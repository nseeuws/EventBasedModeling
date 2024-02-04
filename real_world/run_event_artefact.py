from typing import Callable

import argparse
import numpy as np 
import pandas as pd 
import tensorflow as tf
import logging

import data_artefact
import encoding
import network
import utils
import scoring_utils
import losses


def evaluation_function(
    signals, durations, network,
    stride, down_factor, thresholds,
    max_duration
):
    fs = 200
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
            x = signal[np.newaxis, :size, np.newaxis]
            with tf.device('/cpu:0'):
                pred_loc, pred_size, _ = network.predict(x, verbose=0)
            pred_size = pred_size * fs * max_duration

            ref_centers = np.argwhere(dur_map)
            ref_obj = []
            for center in ref_centers:
                start = int(stride * center - dur_map[center] / 2.)
                stop = int(stride * center + dur_map[center] / 2.)
                ref_obj.append([start, stop])

            for i_thresh, thresh in enumerate(thresholds):
                hyp_obj = encoding.decode_artefact_events(
                    loc=pred_loc, size=pred_size,
                    thresh=thresh, stride=stride
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
    data_path: str, network_path:str,
    duration_threshold: int,
    stride: int, down_factor: int,
):
    fs = 200
    # Network
    model = network.get_event_artefact(input_duration=None)
    model.load_weights(network_path)


    training_targets, val_targets = prep_data(
        data_path=data_path, duration_threshold=duration_threshold, fs=fs, stride=stride
    )
    signals_val, locations_val, durations_val = val_targets

    scale_factor = duration_threshold * fs 
    for i, duration in enumerate(durations_val):
        durations_val[i] = duration * scale_factor


    # EVALUATION
    n_thresholds = 50
    thresholds = np.linspace(start=1e-3, stop=1, num=n_thresholds)
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals_val, durations=durations_val,
        network=model, stride=stride, down_factor=down_factor,
        thresholds=thresholds, max_duration=duration_threshold
    )

    f1_iou = np.nanargmax(scoring_utils.compute_f1_score(*iou))
    f1_ovlp = np.nanargmax(scoring_utils.compute_f1_score(*ovlp))

    return f1_iou, f1_ovlp


def evaluate_event(
    data_path: str, network_path:str,
    duration_threshold: int,
    stride: int, down_factor: int,
    iou_index: int, ovlp_index: int
):
    fs = 200
    # Network
    model = network.get_event_artefact(input_duration=None)
    model.load_weights(network_path)

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
    n_thresholds = 50
    thresholds = np.linspace(start=1e-3, stop=1, num=n_thresholds)
    ## Actual evaluation
    iou, ovlp = evaluation_function(
        signals=signals, durations=durations,
        network=model, stride=stride, down_factor=down_factor,
        thresholds=thresholds, max_duration=duration_threshold
    )

    f1_iou = scoring_utils.compute_f1_score(*iou)[iou_index]
    f1_ovlp = scoring_utils.compute_f1_score(*ovlp)[ovlp_index]

    return f1_iou, f1_ovlp





def train_model(
    input_duration: int, stride: int,
    batch_size:int,
    signals_train, locations_train, durations_train,
    signals_val, locations_val, durations_val,
    total_training_steps: int,
    network_path: str
):
    model = network.get_event_artefact(input_duration=input_duration)

    # Losses
    lambda_r = 5.
    alpha = np.float32(2.)
    beta = np.float32(4.)
    regression_loss = losses.regression_loss_tf_iou
    focal_loss = losses.get_focal_loss(alpha=alpha, beta=beta)

    # Generators
    generator = data_artefact.EventGenerator(
        signals=signals_train,
        locations=locations_train, durations=durations_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration//stride, batch_stride=input_duration//(2*stride),
        network_stride=stride
    )
    val_generator = data_artefact.NoAugEventGenerator(
        signals=signals_val,
        durations=durations_val, locations=locations_val,
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

    # Training
    n_batches = len(generator)
    n_epochs = total_training_steps // n_batches

    best_loss = 1e20
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_focal_avg = tf.keras.metrics.Mean()
    epoch_regr_avg = tf.keras.metrics.Mean()

    @tf.function 
    def regression_step(reg_map, pred_size, f_loss):
        r_loss = regression_loss(
            target_size=reg_map, pred_size=pred_size
        ) / max(1., n_objects)
        loss = f_loss + lambda_r * r_loss
        epoch_regr_avg(r_loss)
        return loss

    @tf.function
    def train_step(x, loc_map, reg_map, n_objects):
        with tf.GradientTape() as t:
            pred_loc, pred_size, pred_logit = model(x, training=True)
            f_loss = focal_loss(
                loc_map=loc_map, pred_loc=pred_loc,
                pred_logit=pred_logit
            ) / tf.math.maximum(1., n_objects)

            loss = tf.cond(
                n_objects > 0,
                lambda: regression_step(
                    reg_map=reg_map, pred_size=pred_size, f_loss=f_loss
                ),
                lambda: f_loss,
            )

            #if n_objects > 0:
            #    r_loss = regression_loss(
            #        target_size=reg_map, pred_size=pred_size
            #    ) / max(1., n_objects)
            #    loss = f_loss + lambda_r * r_loss
            #    epoch_regr_avg(r_loss)
            #else:
            #    loss = f_loss
        epoch_focal_avg(f_loss)
        epoch_loss_avg(loss)

        grad = t.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grad, model.trainable_variables)) 

    # Actual training
    utils.set_tf_loglevel(logging.ERROR)
    for _ in range(n_epochs):
        ## Training loop
        for batch in range(len(generator)):
            x, loc_map, reg_map, n_objects = generator[batch]
            train_step(x, loc_map, reg_map, n_objects)

            #with tf.GradientTape() as t:
            #    pred_loc, pred_size, pred_logit = eventnet(x, training=True)
            #    f_loss = focal_loss(
            #        loc_map=loc_map, pred_loc=pred_loc,
            #        pred_logit=pred_logit
            #    ) / max(1., n_objects)

            #    if n_objects > 0:
            #        r_loss = regression_loss(
            #            target_size=reg_map, pred_size=pred_size
            #        ) / max(1., n_objects)
            #        loss = f_loss + lambda_r * r_loss
            #        epoch_regr_avg(r_loss)
            #    else:
            #        loss = f_loss
            #epoch_focal_avg(f_loss)
            #epoch_loss_avg(loss)

            #grad = t.gradient(loss, eventnet.trainable_variables)
            #optimizer.apply_gradients(zip(grad, eventnet.trainable_variables))
        ## Cleanup
        generator.on_epoch_end()
        epoch_loss_avg.reset_states()
        epoch_focal_avg.reset_states()
        epoch_regr_avg.reset_states()

        ## Validation loop
        for batch in range(len(val_generator)):
            x, loc_map, reg_map, n_objects = val_generator[batch]
            pred_loc, pred_size, pred_logit = model.predict(x, verbose=0)

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
    return None


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
    # Target maps
    locations_train, durations_train = encoding.get_target_maps(
        labels=labels_train, stride=stride, log=False
    )
    locations_val, durations_val = encoding.get_target_maps(
        labels=labels_val, stride=stride, log=False
    )
    # Rescaling
    scale_factor = duration_threshold * fs
    for i, duration in enumerate(durations_train):
        durations_train[i] = duration / scale_factor
    for i, duration in enumerate(durations_val):
        durations_val[i] = duration / scale_factor

    training_targets = (signals_train, locations_train, durations_train)
    val_targets = (signals_val, locations_val, durations_val)

    return training_targets, val_targets


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
    signals_train, locations_train, durations_train = training_targets
    signals_val, locations_val, durations_val = val_targets

    batch_size = 16
    rng = np.random.default_rng(seed=1234)
    #n_epochs = 50
    n_epochs = 100

    generator = data_artefact.EventGenerator(
        signals=signals_train,
        locations=locations_train, durations=durations_train,
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
        #for size_map in durations_train[idx]:
        #    n_events += np.count_nonzero(size_map)
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
            #signals_train=signals_train[idx],
            #locations_train=locations_train[idx], durations_train=durations_train[idx],
            signals_train=iteration_signals, locations_train=iteration_locations,
            durations_train=iteration_durations,
            signals_val=signals_val, locations_val=locations_val, durations_val=durations_val,
            network_path=network_path
        )

        print('Evaluating')
        # Evaluate EventNet
        iou_index, ovlp_index = find_index(
            data_path=data_path, network_path=network_path,
            duration_threshold=duration_threshold,
            stride=stride, down_factor=down_factor
        )
        f1_iou, f1_ovlp = evaluate_event(
            data_path=data_path, network_path=network_path,
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
        df.to_csv(log_path, sep=',', header=None, mode='a') # type: ignore


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