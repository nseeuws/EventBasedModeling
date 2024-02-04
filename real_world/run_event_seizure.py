import argparse 
import pandas as pd
import sklearn
import numpy as np
import logging 
import tensorflow as tf 
import scipy.signal

import data_seizure
import encoding
import utils 
import network 
import losses 
import scoring_utils


def evaluate_event(
    data_path: str, network_path: str,
    duration_threshold: int,
    stride: int, down_factor: int, iou_index: int, ovlp_index: int
):  
    fs = 200

    # Network
    model = network.get_event_seizure(input_duration=None)
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
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals_purged, durations=durations,
        network=model, stride=stride, down_factor=down_factor,
        thresholds=thresholds, duration_cutoff=duration_threshold * fs
    )
    f1_iou = scoring_utils.compute_f1_score(*iou)[iou_index]
    f1_ovlp = scoring_utils.compute_f1_score(*ovlp)[ovlp_index]

    return f1_iou, f1_ovlp

def get_eventnet_events(loc, size, thresh, stride, duration_cutoff):
    """Decode EventNet events

    Args:
        loc (list): Centerpoint predictions
        size (list): Duration predictions
        thresh (float): Cut-off threshold for peak detection
        stride (float): Network output stride

    Returns:
        list: List of decoded events
    """
    peaks, _ = scipy.signal.find_peaks(loc[0, :, 0, 0], width=1, height=thresh)

    hyp_conf = []
    hyp_obj = []

    for peak in peaks:
        duration_prediction = size[0, peak, 0, 0]
        duration_prediction *= duration_cutoff
        #duration_prediction = size[0, peak, 0]
        start = stride * peak - duration_prediction/2.
        stop = stride * peak + duration_prediction/2.
        hyp_conf.append(loc[0, peak, 0, 0])
        hyp_obj.append([start, stop])

    if len(hyp_obj)>0:
            hyp_tensor = np.zeros(shape=(len(hyp_obj), 4), dtype=np.float32)
            hyp_tensor[:, 1] = 0
            hyp_tensor[:, 3] = 1
        
            for i in range(len(hyp_obj)):
                hyp_tensor[i, 0] = hyp_obj[i][0]
                hyp_tensor[i, 2] = hyp_obj[i][1]
                idx = tf.image.non_max_suppression(boxes=hyp_tensor, scores=hyp_conf,
                                            max_output_size=len(hyp_tensor), iou_threshold=0.5)
            hyp_obj = np.asarray(hyp_obj)[idx, :]
    return hyp_obj 

def evaluation_function(
    signals, durations,
    network: tf.keras.Model, stride: int, 
    down_factor:int, thresholds: np.ndarray, duration_cutoff: int
):
    iou_threshold = 0.5

    hit_iou = np.zeros(shape=(len(thresholds),))
    miss_iou = np.zeros(shape=(len(thresholds),))
    fa_iou = np.zeros(shape=(len(thresholds),))
    hit_ovlp = np.zeros(shape=(len(thresholds),))
    miss_ovlp = np.zeros(shape=(len(thresholds),))
    fa_ovlp = np.zeros(shape=(len(thresholds),))

    for signal, dur_map in zip(signals, durations):
        if len(signal) > down_factor:
            size = (len(signal) // down_factor) * down_factor
            x = signal[np.newaxis, :size, :, np.newaxis]
            with tf.device('/cpu:0'):
                pred_loc, pred_size, _ = network.predict(x)
            ref_centers = np.argwhere(dur_map)
            ref_obj = []
            for center in ref_centers:
                start = stride * center  - dur_map[center]/2.
                stop = stride * center  + dur_map[center]/2.
                ref_obj.append([start, stop]) 
            
            for i_thresh, thresh in enumerate(thresholds):
                hyp_obj = get_eventnet_events(
                    loc=pred_loc, size=pred_size,
                    thresh=thresh, stride=stride, duration_cutoff=duration_cutoff
                )
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

def find_index(
    data_path: str, network_path: str,
    duration_threshold: int,
    stride: int, down_factor: int, input_duration: int
):  
    fs = 200

    # Network
    model = network.get_event_seizure(input_duration=None)
    model.load_weights(network_path)


    training_targets, val_targets = prep_data(
        training_path=data_path, duration_threshold=duration_threshold,
        fs=fs, stride=stride, input_duration=input_duration
    )
    signals_val, locations_val, durations_val = val_targets

    # Data

    # Evaluation
    n_thresholds = 40
    thresholds = np.linspace(start=1e-2, stop=1, num=n_thresholds)
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals_val, durations=durations_val,
        network=model, stride=stride, down_factor=down_factor,
        thresholds=thresholds, duration_cutoff=duration_threshold * fs
    )
    iou_index = np.nanargmax(np.nan_to_num(scoring_utils.compute_f1_score(*iou)))
    ovlp_index = np.nanargmax(np.nan_to_num(scoring_utils.compute_f1_score(*ovlp)))

    return iou_index, ovlp_index

def train_model(
    input_duration: int, stride: int,
    batch_size: int, total_training_steps: int,
    signals_train: np.ndarray, 
    locations_train: np.ndarray, durations_train: np.ndarray,
    signals_val: np.ndarray, locations_val: np.ndarray, durations_val: np.ndarray,
    network_path: str
):
    model = network.get_event_seizure(input_duration=input_duration)

    # Losses
    lambda_r = 5.
    alpha = np.float32(2.)
    beta = np.float32(4.)
    regression_loss = losses.regression_loss_tf_iou
    focal_loss = losses.get_focal_loss(alpha=alpha, beta=beta)

    ## Data generators
    generator = data_seizure.EventGenerator(
        signals=signals_train, locations=locations_train, durations=durations_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
    )
    val_generator = data_seizure.EventGenerator(
        signals=signals_val, locations=locations_val, durations=durations_val,
        batch_size=batch_size, shuffle=False,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
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
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_focal_avg = tf.keras.metrics.Mean()
    epoch_regr_avg = tf.keras.metrics.Mean()

    # Actual training
    utils.set_tf_loglevel(logging.ERROR)
    for _ in range(n_epochs):
        ## Training loop
        for batch in range(len(generator)):
            x, loc_map, reg_map, n_objects = generator[batch]

            with tf.GradientTape() as t:
                pred_loc, pred_size, pred_logit = model(x, training=True)
                f_loss = focal_loss(
                    loc_map=loc_map, pred_loc=pred_loc,
                    pred_logit=pred_logit
                ) / max(1., n_objects)

                if n_objects > 0:
                    r_loss = regression_loss(
                        target_size=reg_map, pred_size=pred_size
                    ) / max(1., n_objects)
                    loss = f_loss + lambda_r * r_loss
                    epoch_regr_avg(r_loss)
                else:
                    loss = f_loss
            epoch_focal_avg(f_loss)
            epoch_loss_avg(loss)

            grad = t.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
        ## Cleanup
        generator.on_epoch_end()
        epoch_loss_avg.reset_states()
        epoch_focal_avg.reset_states()
        epoch_regr_avg.reset_states()

        ## Validation loop
        for batch in range(len(val_generator)):
            x, loc_map, reg_map, n_objects = val_generator[batch]
            pred_loc, pred_size, pred_logit = model.predict(x)

            f_loss = focal_loss(
                loc_map=loc_map, pred_loc=pred_loc, pred_logit=pred_logit
            ) / max(1., n_objects)

            if n_objects > 0:
                r_loss = regression_loss(
                    target_size=reg_map, pred_size=pred_size
                ) / max(1., n_objects)
                loss = f_loss + lambda_r * r_loss
                epoch_regr_avg(r_loss)
            else:
                loss = f_loss
            epoch_loss_avg(loss)
            epoch_focal_avg(f_loss)

        ## Storing network weights
        epoch_loss = epoch_loss_avg.result()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.save_weights(network_path)

        ## Cleanup
        epoch_loss_avg.reset_states()
        epoch_focal_avg.reset_states()
        epoch_regr_avg.reset_states()
    utils.set_tf_loglevel(logging.WARNING)

def prep_data(
    training_path: str, duration_threshold: int,
    fs: int, stride: int, input_duration: int
):
    scaled_threshold = duration_threshold * fs
    # Load from disk
    files_training, signals_training, labels_training = data_seizure.load_eeg(path=training_path)

    # Throw out short recordings and long training events
    files_purged, signals_purged, labels_purged = data_seizure.purge_recordings(
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

    # Prep targets
    locations_train, durations_train = encoding.get_target_maps(
        labels=labels_train, stride=stride, log=False
    )
    locations_val, durations_val = encoding.get_target_maps(
        labels=labels_val, stride=stride, log=False
    )

    # Rescale targets
    for i, duration in enumerate(durations_train):
        durations_train[i] = duration / scaled_threshold
    for i, duration in enumerate(durations_val):
        durations_val[i] = duration / scaled_threshold

    training_targets = (signals_train, locations_train, durations_train)
    val_targets = (signals_val, locations_val, durations_val)

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
    signals_train, locations_train, durations_train = training_targets
    signals_val, locations_val, durations_val = val_targets

    batch_size = 16
    rng = np.random.default_rng(seed=1234)
    n_epochs = 100

    generator = data_seizure.EventGenerator(
        signals=signals_train, locations=locations_train, durations=durations_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
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
            n_events += np.count_nonzero(durations_train[index])
        
        print('Training')
        # Train EventNet
        iteration_signals = []
        iteration_locations = []
        iteration_durations = []
        for index in idx:
            iteration_signals.append(signals_train[index])
            iteration_locations.append(locations_train[index])
            iteration_durations.append(durations_train[index])
        train_model(
            input_duration=input_duration, stride=stride,
            batch_size=batch_size, total_training_steps=total_training_steps,
            signals_train=iteration_signals,
            locations_train=iteration_locations, durations_train=iteration_durations,
            signals_val=signals_val, locations_val=locations_val, durations_val=durations_val,
            network_path=network_path
        )

        print('Evaluating')
        # Evaluate EventNet
        iou_index, ovlp_index = find_index(
            data_path=training_path, network_path=network_path,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor, input_duration=input_duration
        )
        f1_iou, f1_ovlp = evaluate_event(
            data_path=test_path, network_path=network_path,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor,
            iou_index=iou_index, ovlp_index=ovlp_index
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