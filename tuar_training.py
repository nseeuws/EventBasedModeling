import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd

import eventnet


def main(args):
    # Get path command line arguments
    data_path = args.data_path
    if args.network_path:
        network_path = args.network_path
    else:
        network_path = 'tuar_eventnet.h5'
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = 'tuar_eventnet_log.csv'

    # Sanity check - check for access to all the files
    assert os.path.isfile(data_path)
    assert os.access(network_path, mode=os.F_OK)
    assert os.access(log_path, mode=os.F_OK)

    # Get training command line arguments
    batch_size = args.batch_size
    learning_rate = args.lr
    n_epochs = args.n_epochs
    duration_factor = args.duration_factor

    # Get loss command line arguments
    lambda_r = args.lambda_r

    # Get network command line arguments
    duration_threshold = args.duration_threshold

    # Command line arguments sanity checks
    assert batch_size > 0
    assert learning_rate > 0.
    assert n_epochs > 0.
    assert lambda_r >= 0.  # You _can_ ignore duration loss if you want to
    assert duration_factor > 0
    assert duration_threshold > 0.

    # ========== ACTUAL SCRIPT ==========
    # Standard parameters (changing these will involve changing implementations as well
    fs = 200
    stride = 4**2

    stride_factor = 4
    n_steps = 6
    down_factor = stride_factor ** n_steps
    input_duration = duration_factor * down_factor

    # DATA
    # Load and prepare data
    training_targets, val_targets = eventnet.data.load_tuar_training_data(
        data_path=data_path, duration_threshold=duration_threshold,
        fs=fs, stride=stride
    )
    signals_train, centers_train, durations_train = training_targets
    signals_val, centers_val, durations_val = val_targets

    # Set up generators to produce training (and validation) batches
    generator = eventnet.training.TUARGenerator(
        signals=signals_train, centers=centers_train, durations=durations_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration // stride, batch_stride=input_duration // (2 * stride),
        network_stride=stride
    )
    val_generator = eventnet.training.TUARGenerator(
        signals=signals_val, centers=centers_val, durations=durations_val,
        batch_size=batch_size, shuffle=False,
        window_size=input_duration // stride, batch_stride=input_duration // (2 * stride),
        network_stride=stride
    )

    # LEARNING SETUP
    # Network
    network = eventnet.network.build_artefact_eventnet(
        input_duration=input_duration
    )

    # Losses
    regression_loss = eventnet.losses.iou_loss
    focal_loss = eventnet.losses.build_focal_loss()

    # Optimizer
    n_batches = len(generator)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=n_epochs // 10 * n_batches,
        decay_rate=0.5,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Bookkeeping
    loss_train = np.zeros(shape=(n_epochs,))
    f_loss_train = np.zeros(shape=(n_epochs,))
    r_loss_train = np.zeros(shape=(n_epochs,))
    loss_val = np.zeros(shape=(n_epochs,))
    f_loss_val = np.zeros(shape=(n_epochs,))
    r_loss_val = np.zeros(shape=(n_epochs,))
    best_loss = 1e20
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_focal_avg = tf.keras.metrics.Mean()
    epoch_regr_avg = tf.keras.metrics.Mean()

    # TRAINING
    for epoch in range(n_epochs):
        print(f'===== Epoch {epoch} =====')

        # Training loop
        for batch in range(len(generator)):
            # Generate current training batch
            signal, center, duration, n_objects = generator[batch]

            # Use TF tape to compute loss gradients
            with tf.GradientTape() as tape:
                # Forward pass
                pred_center, pred_duration, pred_logit = network(signal, training=True)
                # Focal loss, training the center prediction
                f_loss = focal_loss(
                    map_target=center, map_pred=pred_center, logit_pred=pred_logit
                ) / max(1., n_objects)

                # If there are indeed events, also compute duration loss
                if n_objects > 0:
                    r_loss = regression_loss(
                        dur_target=duration, dur_pred=pred_duration
                    ) / max(1., n_objects)
                    loss = f_loss + lambda_r * r_loss  # Combine loss terms
                    epoch_regr_avg(r_loss)  # Keep track of the duration loss
                else:
                    loss = f_loss
            # Backward pass
            grad = tape.gradient(loss, network.trainable_variables)
            # Update network weights
            optimizer.apply_gradients(zip(grad, network.trainable_variables))

            # Make sure we keep track of the different loss terms
            epoch_focal_avg(f_loss)
            epoch_loss_avg(loss)

        # End-of-epoch cleanup
        loss_train[epoch] = epoch_loss_avg.result()
        f_loss_train[epoch] = epoch_focal_avg.result()
        r_loss_train[epoch] = epoch_regr_avg.result()
        print(f'Loss Train ----- {loss_train[epoch]:.4f}')
        print(f'Focal:           {f_loss_train[epoch]:.4f}')
        print(f'Regression:      {r_loss_train[epoch]:.4f}')
        generator.on_epoch_end()
        epoch_loss_avg.reset_states()
        epoch_focal_avg.reset_states()
        epoch_regr_avg.reset_states()

        # Validation loop
        for batch in range(len(val_generator)):
            signal, center, duration, n_objects = val_generator[batch]
            pred_center, pred_duration, pred_logit = network.predict(signal)
            f_loss = focal_loss(
                map_target=center, map_pred=pred_center, logit_pred=pred_logit
            ) / max(1., n_objects)

            if n_objects > 0:
                r_loss = regression_loss(
                    dur_target=duration, dur_pred=pred_duration
                ) / max(1., n_objects)
                loss = f_loss + lambda_r * r_loss
                epoch_regr_avg(r_loss)
            else:
                loss = f_loss
            epoch_focal_avg(f_loss)
            epoch_loss_avg(loss)

        loss_val[epoch] = epoch_loss_avg.result()
        f_loss_val[epoch] = epoch_focal_avg.result()
        r_loss_val[epoch] = epoch_regr_avg.result()
        epoch_loss_avg.reset_states()
        epoch_focal_avg.reset_states()
        epoch_regr_avg.reset_states()
        # Storing network weights
        if loss_val[epoch] < best_loss:
            best_loss = loss_val[epoch]
            network.save_weights(network_path)
        print(f'Loss Val   ----- {loss_val[epoch]:.4f}')
        print(f'Focal:           {f_loss_val[epoch]:.4f}')
        print(f'Regression:      {r_loss_val[epoch]:.4f}')

    # Store loss logs
    df = pd.DataFrame(data={
        'loss_train': loss_train,
        'center_loss_train': f_loss_train,
        'duration_loss_train': r_loss_train,
        'loss_val': loss_val,
        'center_loss_val': f_loss_val,
        'duration_loss_val': r_loss_val
    })
    df.to_csv(log_path)


def parser_builder():
    parser = argparse.ArgumentParser('Options')

    # Paths
    parser.add_argument(
        '--data_path', type=str,
        help="Path to the TUAR HDF5 storage object",
        required=True
    )
    parser.add_argument(
        '--network_path', type=str,
        help="Optional, path to where the EventNet network weights should be stored."
    )
    parser.add_argument(
        '--log_path', type=str,
        help="Optional, path to where to store loss logs."
    )

    # Training details
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help="Batch size"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help="Base learning rate. Will decay throughout training"
    )
    parser.add_argument(
        '--n_epochs', type=int, default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        '--duration_factor', type=int, default=10,
        help="How large should the training window be? Recommended to not change"
    )

    # Losses
    parser.add_argument(
        '--lambda_r', type=float, default=5.,
        help="Relative regression loss weight."
    )

    # Network details
    parser.add_argument(
        '--duration_threshold', type=float, default=10.,
        help='Maximum duration for EventNet'
    )

    return parser


if __name__ == '__main__':
    argument_parser = parser_builder()
    arguments = argument_parser.parse_args()
    main(args=arguments)
