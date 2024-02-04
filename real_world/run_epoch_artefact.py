import argparse
import numpy as np 
import pandas as pd 
import tensorflow as tf
import logging 
import scipy.signal

import data_artefact
import encoding
import network
import utils
import scoring_utils


def evaluation_function(
    signals, labels, network,
    stride, down_factor, thresholds,
):
    fs = 200
    iou_threshold = 0.5

    hit_iou = np.zeros(shape=(len(thresholds),))
    miss_iou = np.zeros(shape=(len(thresholds),))
    fa_iou = np.zeros(shape=(len(thresholds),))
    hit_ovlp = np.zeros(shape=(len(thresholds),))
    miss_ovlp = np.zeros(shape=(len(thresholds),))
    fa_ovlp = np.zeros(shape=(len(thresholds),))

    for signal, label in zip(signals, labels):
        if len(signal) > down_factor:
            size = (len(signal) // down_factor) * down_factor
            x = signal[np.newaxis, :size, np.newaxis]
            with tf.device('/cpu:0'):
                label_pred = network.predict(x)

            ref_obj = ysyw_utils.get_objects(label)

            time_pp = int(0.1 * fs / stride)
            if time_pp % 2 == 0:
                time_pp += 1

            for i_thresh, thresh in enumerate(thresholds):
                y_thresh = np.asarray(label_pred[0, :, 0] >= thresh, dtype=np.int8)
                y_filt = scipy.signal.medfilt(volume=y_thresh, kernel_size=time_pp)

                hyp_obj_raw = encoding.get_objects(y_filt)
                hyp_obj = []
                for event in hyp_obj_raw:
                    hyp_obj.append([int(stride * event[0]), int(stride * event[1])])

                tp_ovlp, fn_ovlp, fp_ovlp = scoring_utils.ovlp_score(ref_obj, hyp_obj)

                iou_output = scoring_utils.performance_evaluation(
                    ref_obj=ref_obj, hyp_obj=hyp_obj,
                    iou_threshold=iou_threshold,
                )
                tp_iou, fn_iou, fp_iou = iou_output

                hit_iou[i_thresh] += tp_iou
                miss_iou[i_thresh] += fn_iou
                fa_iou[i_thresh] += fp_iou

                hit_ovlp[i_thresh] += tp_ovlp
                miss_ovlp[i_thresh] += fn_ovlp
                fa_ovlp[i_thresh] += fp_ovlp

    prec_iou = scoring_utils.get_precision(
        tp=hit_iou, fn=miss_iou, fp=fa_iou
    )
    rec_iou = scoring_utils.get_recall(
        tp=hit_iou, fn=miss_iou, fp=fa_iou
    )
    prec_ovlp = scoring_utils.get_precision(
        tp=hit_ovlp, fn=miss_ovlp, fp=fa_ovlp
    )
    rec_ovlp = scoring_utils.get_recall(
        tp=hit_ovlp, fn=miss_ovlp, fp=fa_ovlp
    )
    return (prec_iou, rec_iou), (prec_ovlp, rec_ovlp)

def prep_data(
    data_path: str, duration_threshold: int, fs: int, stride: int
):
    # Load data
    signals_train, labels_train, signals_val, labels_val = data_artefact.get_training_data(
        data_path=data_path
    )
    # Purge
    signals_train, labels_train = data_artefact.filter_data(
        signals=signals_train, labels=labels_train,
        duration_threshold=duration_threshold, fs=fs
    )
    signals_val, labels_val = data_artefact.filter_data(
        signals=signals_val, labels=labels_val,
        duration_threshold=duration_threshold, fs=fs
    )
    ## Downsample labels
    labels_train = data_artefact.downsample(labels=labels_train, stride=stride)
    labels_val = data_artefact.downsample(labels=labels_val, stride=stride)

    training_targets = (signals_train, labels_train)
    val_targets = (signals_val, labels_val)

    return training_targets, val_targets


def train_unet(
    input_duration: int, stride: int,
    batch_size:int,
    signals_train, labels_train,
    signals_val, labels_val,
    total_training_steps: int,
    network_path: str
):
    unet = network.get_epoch_artefact(input_duration=input_duration)

    # Losses
    bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)
    #event_factor = 100.
    event_factor = np.float32(100.)


    # Generators
    generator = data_artefact.EpochGenerator(
        signals=signals_train,
        labels=labels_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
    )
    val_generator = data_artefact.NoAugEpochGenerator(
        signals=signals_val, labels=labels_val,
        batch_size=batch_size, shuffle=False,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
    )

    # Optimizer
    #lr=5e-4
    lr = 1e-3
    optimizer = tf.keras.optimizers.Adam
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=total_training_steps // 10,
        decay_rate=0.5,
        staircase=True
    )
    optimizer = optimizer(learning_rate=lr_schedule)

    # Training
    n_batches = len(generator)
    n_epochs = total_training_steps // n_batches

    best_loss = 1e20
    epoch_loss_avg = tf.keras.metrics.Mean()

    # Actual training
    
    utils.set_tf_loglevel(logging.ERROR)
    for _ in range(n_epochs):
        ## Training loop
        for batch in range(len(generator)):
            x, y = generator[batch]
            sample_weight = np.where(y, event_factor, 1.)

            with tf.GradientTape() as t:
                y_pred = unet(x, training=True)
                loss = bce(y_true=y, y_pred=y_pred, sample_weight=sample_weight)
            epoch_loss_avg(loss)

            grad = t.gradient(loss, unet.trainable_variables)
            optimizer.apply_gradients(zip(grad, unet.trainable_variables))
        ## Cleanup
        generator.on_epoch_end()
        epoch_loss_avg.reset_states()

        ## Validation loop
        for batch in range(len(val_generator)):
            x, y = val_generator[batch]
            sample_weight = np.where(y, event_factor, 1.)
            y_pred = unet.predict(x)
            loss = bce(y_true=y, y_pred=y_pred, sample_weight=sample_weight)
            epoch_loss_avg(loss)

        loss_val = epoch_loss_avg.result()
        ## Cleanup
        epoch_loss_avg.reset_states()

        ## Storing network weights
        if loss_val < best_loss:
            best_loss = loss_val
            unet.save_weights(network_path)
    
    utils.set_tf_loglevel(logging.WARNING)

    return None

def find_index(
    data_path: str, network_path:str,
    duration_threshold: int,
    stride: int, down_factor: int,
):
    fs = 200
    model = network.get_epoch_artefact(input_duration=None)
    model.load_weights(network_path)


    training_targets, val_targets = prep_data(
        data_path=data_path, duration_threshold=duration_threshold, fs=fs, stride=stride
    )
    signals_val, labels_val = val_targets


    # EVALUATION
    n_thresholds = 40
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals_val, labels=labels_val,
        network=model, stride=stride, down_factor=down_factor,
        thresholds=thresholds 
    )

    f1_iou = np.nanargmax(scoring_utils.compute_f1_score(*iou))
    f1_ovlp = np.nanargmax(scoring_utils.compute_f1_score(*ovlp))

    return f1_iou, f1_ovlp


def evaluate_unet(
    data_path: str, network_path:str,
    duration_threshold: int,
    stride: int, down_factor: int,
    iou_index:int, ovlp_index: int
):
    fs = 200
    # Network
    unet = network.get_epoch_artefact(input_duration=None)
    unet.load_weights(network_path)

    # Load data
    signals, labels = data_artefact.get_testing_data(data_path)
    ## Purge
    signals, labels = data_artefact.filter_data(
        signals=signals, labels=labels, duration_threshold=duration_threshold, fs=fs
    )
    ## Target maps
    _, durations = encoding.get_target_maps(
        labels=labels, stride=stride, log=False
    )

    # EVALUATION
    n_thresholds = 40
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals, labels=labels,
        network=unet, stride=stride, down_factor=down_factor,
        thresholds=thresholds
    )

    f1_iou = scoring_utils.compute_f1_score(*iou)[iou_index]
    f1_ovlp = scoring_utils.compute_f1_score(*ovlp)[ovlp_index]

    return f1_iou, f1_ovlp


def train_n_recordings(
    n_recordings: list[int],
    data_path: str, duration_threshold: int,
    base_path: str, log_path:str,
    input_duration: int, stride: int, down_factor: int
):
    fs = 200
    training_targets, val_targets = prep_data(
        data_path=data_path, duration_threshold=duration_threshold, fs=fs, stride=stride
    )
    signals_train, labels_train = training_targets
    signals_val, labels_val = val_targets

    batch_size = 16
    rng = np.random.default_rng(seed=1234)
    #n_epochs = 50
    n_epochs = 100

    generator = data_artefact.EpochGenerator(
        signals=signals_train, labels=labels_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
    )
    total_training_steps = n_epochs * len(generator)

    for n_record in n_recordings:
        print(f'===== {n_record} recordings =====')
        assert n_record <= len(signals_train)
        idx = rng.choice(a=len(signals_train), size=n_record, replace=False)

        # Define network path
        network_path = f'{base_path}/tuar_network_{n_record}_recordings.h5'

        # Count events
        n_events = 0
        for index in idx:
            events = encoding.get_objects(label_array=labels_train[index])
            n_events += len(events)

        print('Training')
        # Train EventNet

        iteration_signals = []
        iteration_labels = []

        for index in idx:
            iteration_signals.append(signals_train[index])
            iteration_labels.append(labels_train[index])

        train_unet(
            input_duration=input_duration, stride=stride,
            batch_size=batch_size, total_training_steps=total_training_steps,
            signals_train=iteration_signals, labels_train=iteration_labels,
            signals_val=signals_val, labels_val=labels_val,
            network_path=network_path
        )

        print('Evaluating')
        # Evaluate EventNet
        iou_index, ovlp_index = find_index(
            data_path=data_path, network_path=network_path,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor
        )
        f1_iou, f1_ovlp = evaluate_unet(
            data_path=data_path, network_path=network_path,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor,
            iou_index=iou_index, ovlp_index=ovlp_index
        )
        print(f'IoU:  {f1_iou}')
        print(f'OVLP: {f1_ovlp}')

        # Save everything
        df = pd.DataFrame(data={
            'n_events':[n_events, ],
            'F1_IoU': [f1_iou, ],
            'F1_OVLP': [f1_ovlp, ],
            'network': [network_path, ]
        })
        df.to_csv(log_path, sep=',', header=None, mode='a')


def main(args):
    n_recordings = [args.n_recordings,]
    data_path = args.data_path
    log_path = args.log_path
    base_path = args.base_path

    duration_threshold = 10

    stride = 4**2
    n_steps = 6
    stride_factor = 4
    down_factor = stride_factor**n_steps
    duration_factor = 10
    input_duration = duration_factor*down_factor

    train_n_recordings(
        n_recordings=n_recordings, data_path=data_path,
        log_path=log_path, base_path=base_path,
        duration_threshold=duration_threshold,
        input_duration=input_duration, stride=stride,
        down_factor=down_factor
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Script options')

    parser.add_argument(
        '--n_recordings', type=int
    )
    parser.add_argument(
        '--data_path', type=str
    )
    parser.add_argument(
        '--base_path', type=str
    )
    parser.add_argument(
        '--log_path', type=str
    )

    args = parser.parse_args()
    
    main(args=args)