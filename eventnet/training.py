from typing import List
import numpy as np
import tensorflow as tf


class TUSZGenerator(tf.keras.utils.Sequence):
    def __init__(
            self, signals: np.ndarray, centers: np.ndarray, durations: np.ndarray,
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

        self.n_channels = signals[0].shape

        key_array = []
        for i, array in enumerate(self.locations):
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
        for i, array in enumerate(self.locations):
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
            center_ = self.locations[key[0]][key[1]:key[1] + self.window_size]
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
