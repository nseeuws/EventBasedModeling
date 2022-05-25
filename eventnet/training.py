from typing import List
import numpy as np
import tensorflow as tf
import pandas as pd

import eventnet.losses


class TUSZGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, signals: List[np.ndarray], centers: List[np.ndarray], durations: List[np.ndarray],
            batch_size: int, batch_stride: int, window_size: int,
            network_stride: int, shuffle=True
    ):
        super().__init__()
        self.signals = signals
        self.centers = centers
        self.durations = durations
        self.batch_size = batch_size
        self.stride = batch_stride
        self.network_stride = network_stride
        self.window_size = window_size
        self.shuffle = shuffle

        self.n_channels = signals[0].shape[1]

        key_array = []
        for i, array in enumerate(self.centers):
            n = (len(array) - self.window_size) // self.stride
            for j in range(n):
                key_array.append([i, self.stride * j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __getitem__(self, item):
        keys = np.arange(
            start=item * self.batch_size,
            stop=(item + 1) * self.batch_size
        )

        stride = self.network_stride
        x = np.empty(shape=(
            self.batch_size, self.window_size * stride, self.n_channels, 1
        ), dtype=np.float32)

        center = np.empty(shape=(self.batch_size, self.window_size, 1, 1), dtype=np.float32)
        duration = np.empty(shape=(self.batch_size, self.window_size, 1, 1), dtype=np.float32)

        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            x[i, :, :, 0] = self.signals[key[0]][key[1] * stride:stride * (key[1] + self.window_size), :]
            center[i, :, 0, 0] = self.centers[key[0]][key[1]:key[1] + self.window_size]
            duration[i, :, 0, 0] = self.durations[key[0]][key[1]:key[1] + self.window_size]

        n_objects = np.float32(np.count_nonzero(duration))

        return x, center, duration, n_objects


class TUARGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, signals: List[np.ndarray], centers: List[np.ndarray], durations: List[np.ndarray],
            batch_size: int, batch_stride: int, window_size: int,
            network_stride: int, shuffle=True
    ):
        super().__init__()
        self.signals = signals
        self.centers = centers
        self.durations = durations
        self.batch_size = batch_size
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride
        self.shuffle = shuffle

        self.rng = np.random.default_rng()

        key_array = []
        for i, array in enumerate(self.centers):
            n = (len(array) - self.window_size) // self.stride
            for j in range(n):
                key_array.append([i, self.stride * j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __getitem__(self, item):
        keys = np.arange(
            start=item * self.batch_size,
            stop=(item + 1) * self.batch_size
        )
        stride = self.network_stride
        x = np.empty(shape=(self.batch_size, self.window_size * stride, 1), dtype=np.float32)
        center = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        duration = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        flip = self.rng.integers(low=0, high=1, size=self.batch_size, endpoint=True)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            signal = self.signals[key[0]][key[1] * stride:stride * (key[1] + self.window_size)].T
            center_ = self.centers[key[0]][key[1]:key[1] + self.window_size]
            duration_ = self.durations[key[0]][key[1]:key[1] + self.window_size]
            if flip[i] and self.shuffle:
                signal = np.flip(signal)
                center_ = np.flip(center_)
                duration_ = np.flip(duration_)
            x[i, :, 0] = signal
            center[i, :, 0] = center_
            duration[i, :, 0] = duration_

        if self.shuffle:
            scaling = self.rng.integers(low=0, high=1, size=(self.batch_size, 1, 1), endpoint=True)
            scaling = 2. * scaling - 1.
            x = scaling * x

        n_objects = np.float32(np.count_nonzero(duration))

        return x, center, duration, n_objects


def training_loop(
        network: tf.keras.Model,
        generator: tf.keras.utils.Sequence, val_generator: tf.keras.utils.Sequence,
        learning_rate: float, n_epochs: int, lambda_r: float,
        log_path: str, network_path: str
) -> None:
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
    pass
