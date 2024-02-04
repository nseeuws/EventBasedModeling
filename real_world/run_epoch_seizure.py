from typing import Tuple 
import logging 
import argparse
import numpy as np 
import pandas as pd 
import sklearn.model_selection
import tensorflow as tf 

import utils 
import scoring_utils 
import encoding 
import network 
import data_seizure


def evaluation_function(
    signals, labels,
    network: tf.keras.Model, down_factor: int,
    thresholds: np.ndarray, iou_index:int,
    ovlp_index: int
) -> Tuple[float, float]:
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
            x = signal[np.newaxis, :size, :, np.newaxis]
            with tf.device('/cpu:0'): # type: ignore
                label_pred = network.predict(x)
            prediction = label_pred[0, :, 0, 0]
            
            ref_obj = encoding.get_objects(label)

            for i_thresh, thresh in enumerate(thresholds):
                hyp_obj = neureka_event_decoding(prediction=prediction, threshold=thresh)
                tp_ovlp, fn_ovlp, fp_ovlp = scoring_utils.ovlp_score(ref_obj, hyp_obj)
                #hyp_obj = simple_decoding(prediction=prediction, threshold=thresh)
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

    f1_iou = scoring_utils.compute_f1_score(
        prec_array=prec_iou,
        rec_array=rec_iou
    )[iou_index]
    f1_ovlp = scoring_utils.compute_f1_score(
        prec_array=prec_ovlp,
        rec_array=rec_ovlp
    )[ovlp_index]

    return f1_iou, f1_ovlp


def find_index(
    signals, labels,
    network: tf.keras.Model, down_factor: int,
    thresholds: np.ndarray
) -> Tuple[int, int]:
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
            x = signal[np.newaxis, :size, :, np.newaxis]
            with tf.device('/cpu:0'): # type: ignore
                label_pred = network.predict(x)
            prediction = label_pred[0, :, 0, 0]
            
            ref_obj = encoding.get_objects(label)

            for i_thresh, thresh in enumerate(thresholds):
                hyp_obj = neureka_event_decoding(prediction=prediction, threshold=thresh)
                tp_ovlp, fn_ovlp, fp_ovlp = scoring_utils.ovlp_score(ref_obj, hyp_obj)
                #hyp_obj = simple_decoding(prediction=prediction, threshold=thresh)
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

    f1_iou = np.nanargmax(scoring_utils.compute_f1_score(
        prec_array=prec_iou,
        rec_array=rec_iou
    ))
    f1_ovlp = np.nanargmax(scoring_utils.compute_f1_score(
        prec_array=prec_ovlp,
        rec_array=rec_ovlp
    ))

    return f1_iou, f1_ovlp # type: ignore


def merge_events(hyp_obj, duration_threshold):
    i = 1
    tot_len = len(hyp_obj)
    while i < tot_len:
        if (hyp_obj[i][0] - hyp_obj[i-1][1]) < duration_threshold:
            hyp_obj[i-1][1] = hyp_obj[i][1]
            hyp_obj.pop(i)
            tot_len -= 1
        else:
            i += 1
    return hyp_obj

def neureka_event_decoding(prediction: np.ndarray, threshold: float) -> list:
    fs = 200
    # Threshold prediction signal
    thresholded_prediction = np.asarray((prediction - np.median(prediction)) > threshold, dtype=np.int8)

    # Get first list of hypothesis events
    hyp_obj = encoding.get_objects(thresholded_prediction)

    # Merge events closer than 30 seconds
    hyp_obj = merge_events(hyp_obj, duration_threshold=30 * fs)

    # Remove events with mean prediction < 82% of event with max prediction
    if len(hyp_obj):
        amp = list()
        for event in hyp_obj:
            amp.append(np.mean(prediction[event[0]:event[1]]))
        amp = np.array(amp)
        amp /= np.max(amp)

        hyp_obj = list(np.array(hyp_obj)[amp > 0.82])
    
    # Remove short events
    final_hyp = []
    for event in hyp_obj:
        #if event[1] - event[0] > 15 * fs:
        if event[1] - event[0] > 5 * fs:
            final_hyp.append([event[0] + fs, event[1] - fs])
    
    return final_hyp


def evaluate_epoch(
    data_path: str, network_path: str,
    duration_threshold: int,
    signals_val, labels_val,
    stride: int, down_factor: int,
):  
    fs = 200

    # Network
    model = network.get_epoch_seizure(input_duration=None) # type: ignore
    model.load_weights(network_path)  
    

    # Data
    ## Load data from disk
    files, signals, labels = data_seizure.load_eeg(data_path)
    ## Purge SHORT recordings (for evaluation we don't discard long seizures)
    _, signals_purged, labels_purged = data_seizure.purge_test_recordings(
        input_duration=down_factor, files_training=files,
        signals_training=signals, labels_training=labels
    )
    ## Extract targets
    _, durations = encoding.get_target_maps(
        labels=labels_purged, stride=stride, log=False
    )

    # Evaluation
    n_thresholds = 40
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)
    iou_index, ovlp_index = find_index(
            signals=signals_val, labels=labels_val,
            network=model, thresholds=thresholds,
            down_factor=down_factor
        )
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals_purged, labels=labels_purged,
        network=model, down_factor=down_factor,
        thresholds=thresholds,
        iou_index=iou_index, ovlp_index=ovlp_index
    )

    return iou, ovlp


def train_unet(
    input_duration: int, stride: int,
    batch_size: int, total_training_steps: int,
    signals_train: np.ndarray, 
    labels_train: np.ndarray,
    signals_val: np.ndarray, labels_val: np.ndarray,
    network_path: str
):
    unet = network.get_epoch_seizure(input_duration=input_duration)

    # Losses
    bin_xent = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.2)

    ## Data generators
    generator = data_seizure.EpochGenerator(
        signals=signals_train, labels=labels_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration, batch_stride=input_duration//2,
    )
    val_generator = data_seizure.EpochGenerator(
        signals=signals_val, labels=labels_val,
        batch_size=batch_size, shuffle=False,
        window_size=input_duration, batch_stride=input_duration//2,
    )

    # Optimizer
    lr=1e-3
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
    one = np.float32(1.)
    epoch_loss_avg = tf.keras.metrics.Mean()

    # Loss weights
    all_labels = np.copy(np.concatenate(labels_train))
    n_bckg = np.sum(all_labels==0)
    n_seiz = np.sum(all_labels==1)
    del all_labels

    # Actual training
    utils.set_tf_loglevel(logging.ERROR)
    for _ in range(n_epochs):
        ## Training loop
        for batch in range(len(generator)):
            x, y = generator[batch]

            with tf.GradientTape() as t:
                y_ = unet(x, training=True)
                loss = tf.reduce_mean(bin_xent(
                    y_true=y, y_pred=y_, 
                    sample_weight=n_bckg / n_seiz * y + (one - y)
                ))
            epoch_loss_avg(loss)

            grad = t.gradient(loss, unet.trainable_variables)
            optimizer.apply_gradients(zip(grad, unet.trainable_variables))
        ## Cleanup
        generator.on_epoch_end()
        epoch_loss_avg.reset_states()

        ## Validation loop
        for batch in range(len(val_generator)):
            x, y = val_generator[batch]
            y_ = unet.predict(x)

            with tf.GradientTape() as t:
                y_ = unet(x, training=True)
                loss = tf.reduce_mean(bin_xent(
                    y_true=y, y_pred=y_, 
                    sample_weight=n_bckg / n_seiz * y + (one - y)
                ))
            epoch_loss_avg(loss)

        ## Storing network weights
        epoch_loss = epoch_loss_avg.result()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            unet.save_weights(network_path)

        ## Cleanup
        epoch_loss_avg.reset_states()
    utils.set_tf_loglevel(logging.WARNING)


def prep_data(
    training_path: str, duration_threshold: int,
    fs: int, stride: int, input_duration: int
):
    scaled_threshold = duration_threshold * fs
    # Load from disk
    files_training, signals_training, labels_training = data_seizure.load_eeg(path=training_path)

    # Throw out short recordings and long training events
    _, signals_purged, labels_purged = data_seizure.purge_recordings(
        input_duration=input_duration, files_training=files_training,
        signals_training=signals_training, labels_training=labels_training,
        threshold=scaled_threshold
    )
    del files_training, signals_training, labels_training

    # Determine what recordings actually contain seizures
    seizure_label = data_seizure.find_seizure_recordings(
        labels=labels_purged
    )

    ## Train-val split
    signals_train, signals_val, labels_train, labels_val = sklearn.model_selection.train_test_split(
        signals_purged, labels_purged, test_size=0.2, random_state=42, stratify=seizure_label
    )

    training_targets = (signals_train, labels_train)
    val_targets = (signals_val, labels_val)

    return training_targets, val_targets


def train_n_recordings(
    n_recordings: list[int],
    training_path: str, test_path: str,
    duration_threshold: int,
    base_path: str, log_path: str,
    input_duration: int, stride: int, down_factor: int
):
    fs = 200
    training_targets, val_targets = prep_data(
        training_path=training_path, duration_threshold=duration_threshold,
        fs=fs, stride=stride, input_duration=input_duration
    )
    signals_train, labels_train = training_targets
    signals_val, labels_val = val_targets

    batch_size = 8
    rng = np.random.default_rng(seed=1234)
    n_epochs = 100

    generator = data_seizure.EpochGenerator(
        signals=signals_train, labels=labels_train, 
        batch_size=batch_size, shuffle=True, 
        batch_stride=input_duration//2, window_size=input_duration
    )

    total_training_steps = n_epochs * len(generator)

    for n_record in n_recordings:
        print(f'===== {n_record} =====')
        assert n_record <= len(signals_train)
        idx = rng.choice(a=len(signals_train), size=n_record, replace=False)

        # Define network path
        network_path = f'{base_path}/tusz_network_{n_record}_recordings.h5'

        # Count events
        n_events = 0
        for index in idx:
            n_events += len(encoding.get_objects(labels_train[index]))
        
        print('Training')
        # Train U-Net
        iteration_signals = []
        iteration_labels = []
        for index in idx:
            iteration_signals.append(signals_train[index])
            iteration_labels.append(labels_train[index])
        train_unet(
            input_duration=input_duration, stride=stride,
            batch_size=batch_size, total_training_steps=total_training_steps,
            signals_train=iteration_signals,
            labels_train=iteration_labels,
            signals_val=signals_val, labels_val=labels_val,
            network_path=network_path
        )

        print('Evaluating')

        # Evaluation
        f1_iou, f1_ovlp = evaluate_epoch(
            data_path=test_path, network_path=network_path,
            signals_val=signals_val, labels_val=labels_val,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor,
        )
        print(f'IoU:  {f1_iou}')
        print(f'OVLP: {f1_ovlp}')

        # Save everything
        df = pd.DataFrame(data={
            'n_events':[n_events,],
            'F1_IoU': [f1_iou,],
            'F1_OVLP': [f1_ovlp,],
            'network':[network_path,]
        })
        df.to_csv(log_path, sep=',', header=None, mode='a')
      

    return None


def main(args):
    n_recordings = [args.n_recordings,]
    training_path = args.training_path
    test_path = args.test_path
    log_path = args.log_path
    base_path = args.base_path

    duration_threshold = 250

    n_steps = 4
    stride_factor = 4
    duration_factor = 10
    stride = stride_factor**n_steps
    down_factor = stride_factor**6
    input_duration = duration_factor * down_factor

    train_n_recordings(
        n_recordings=n_recordings,
        training_path=training_path, test_path=test_path,
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
    #parser.add_argument(
    #    '--data_path', type=str
    #)
    parser.add_argument(
        '--training_path', type=str
    )
    parser.add_argument(
        '--test_path', type=str
    )
    parser.add_argument(
        '--base_path', type=str
    )
    parser.add_argument(
        '--log_path', type=str
    )
    args = parser.parse_args()
    main(args=args)