from collections import Counter, defaultdict
import h5py 
import numpy as np
import random

import tensorflow as tf

import encoding


def downsample(labels, stride):
    for i, label in enumerate(labels):
        labels[i] = label[::stride]
    return labels


def filter_data(signals, labels, duration_threshold=50, fs=200):
    signals_filter = []
    labels_filter = []

    for signal, label in zip(signals, labels):
        events = encoding.get_objects(label_array=label)
        
        if len(events) > 0:
            duration_flag = [(event[1] - event[0]) <= duration_threshold * fs for event in events]

            if all(duration_flag):
                signals_filter.append(signal)
                labels_filter.append(label)
    #signals_filter = np.asarray(signals_filter)
    #labels_filter = np.asarray(labels_filter)
    return signals_filter, labels_filter


def load_data(data_path):
    """Load signal data from the hdfs file

    Args:
        data_path (string): Path to .h5 file containing data

    Returns:
        tuple: Tuple of lists. In order: filenames, signals,
        channel labels, actual labels.
    """
    with h5py.File(data_path, 'r') as f:
        file_names = []
        signals = []
        ch_labels = []
        labels = []
    
        file_names_ds = f['filenames']
        signals_ds = f['signals']
        ch_labels_ds = f['ch_labels']
        labels_ds = f['labels']
    
        for i in range(len(file_names_ds)):
            file_names.append(file_names_ds[i])
            data = np.asarray(np.vstack(signals_ds[i]), dtype=np.float32)
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            signals.append((data - mean) / std)
            ch_labels.append(ch_labels_ds[i])
            labels.append(np.stack(
                [np.vstack(channel_array) for channel_array in labels_ds[i]],
                axis=0))
    
    return file_names, signals, ch_labels, labels


def stack_data_indexed(
    signals, labels, file_names, n_channels, index
):
    """Stack multi-channel data into a single-channel array, based on the label index

    Args:
        signals (list): List of multi-channel EEG signals
        labels (list): List of channel-level labels for the EEG signals
        file_names (list): List of filenames (containing patient ID for each signal entry)
        n_channels (int): Number of channels in the multi-channel EEG
        index (list): Label indices

    Returns:
        tuple: Tuple of lists, containing the "stacked" data.
        In order: signals, labels, patient IDs
    """
    signals_stack = []
    labels_stack = []
    patient_stack = []

    for i in range(len(signals)):
        for i_ch in range(n_channels):
            signals_stack.append(signals[i][i_ch])
            if len(index) == 1:
                labels_stack.append(labels[i][i_ch, index[0], :])
            else:
                labels_stack.append(np.max(labels[i][i_ch, index, :], axis=0))
            patient_stack.append(file_names[i][0:8])
    
    signals_stack = np.asarray(signals_stack, dtype=object)
    labels_stack = np.asarray(labels_stack, dtype=object)
    patient_stack = np.asarray(patient_stack)

    return signals_stack, labels_stack, patient_stack


def stratified_group_k_fold(X, y, groups, k, seed=None):
    """Create stratified, grouped data indices for K-fold cv.

    Args:
        X (list): List of data points
        y (list): List of categorical data labels
        groups (list): List of group IDs
        k (int): Number of folds
        seed (int, optional): Random seed. Defaults to None.

    Returns:
        NoneType: No return values

    Yields:
        Tuple: Tuple of training and testing indices for the current fold. 
    """
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)
    
    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_single_channel_indexed(data_path, CH, index):
    file_names, signals, _, labels = load_data(data_path)
    signals_stack, labels_stack, patient_stack = stack_data_indexed(
        signals=signals, labels=labels, file_names=file_names,
        n_channels=CH, index=index
    )
    # Find signals containing targets
    target_stack = np.asarray([np.sum(labels) > 0 for labels in labels_stack])

    # Create training and testing indices
    fold_splitter = stratified_group_k_fold(
        X=signals_stack, y=target_stack, groups=patient_stack, k=5, seed=1234
    )
    idx, test_idx = next(fold_splitter) # Training data - Test data split
    training_fold_splitter = stratified_group_k_fold(
        X=signals_stack[idx], y=target_stack[idx], groups=patient_stack[idx], k=5, seed=42
    )
    train_idx, val_idx = next(training_fold_splitter) # Training - validation split
    return signals_stack, labels_stack, idx, train_idx, val_idx, test_idx


def get_training_data(data_path):
    CH = 18
    label_dict = {'eyem':0, 'chew':1, 'shiv':2, 'musc':3, 'elec':4}
    index = (label_dict['musc'], label_dict['chew'])
    signals, labels, idx, train_idx, val_idx, _ = get_single_channel_indexed(
        data_path=data_path, CH=CH, index=index
    )
    signals_train = signals[idx][train_idx]
    signals_val = signals[idx][val_idx]
    labels_train = labels[idx][train_idx]
    labels_val = labels[idx][val_idx]
    return signals_train, labels_train, signals_val, labels_val


class EventGenerator(tf.keras.utils.Sequence):
    def __init__(self, signals, locations, durations, batch_size=128, shuffle=True, batch_stride=200, window_size=1024,
                 network_stride=256):
        super().__init__()
        self.signals = signals
        self.locations = locations
        self.durations = durations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride
        self.rng = np.random.default_rng()
        self.n_channels = signals[0].shape[0]
        self.rng = np.random.default_rng()
        key_array = []
        for i, array in enumerate(self.locations):
            n = (len(array) - self.window_size)//self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)
        x, location, duration = self.__data_generation__(keys)
        n_objects = np.float32(np.count_nonzero(duration)) # Approximate method of counting the # of events
        scaling = self.rng.integers(low=0, high=1, size=(self.batch_size, 1, 1), endpoint=True)
        scaling = 2. * scaling - 1.
        x = scaling * x

        return x, location, duration, n_objects

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        stride = self.network_stride
        x = np.empty(shape=(self.batch_size, self.window_size*stride, 1), dtype=np.float32)
        location = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        duration = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        flip = self.rng.integers(low=0, high=1, size=self.batch_size, endpoint=True)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            signal = self.signals[key[0]][key[1]*stride:stride*(key[1]+self.window_size)].T
            location_ = self.locations[key[0]][key[1]:key[1]+self.window_size]
            duration_ = self.durations[key[0]][key[1]:key[1]+self.window_size]
            if flip[i]:
                signal = np.flip(signal)
                location_ = np.flip(location_)
                duration_ = np.flip(duration_)
            x[i, :, 0] = signal
            location[i, :, 0] = location_
            duration[i, :, 0] = duration_
        return x, location, duration
    

class NoAugEventGenerator(tf.keras.utils.Sequence):
    def __init__(self, signals, locations, durations, batch_size=128, shuffle=True, batch_stride=200, window_size=1024,
                 network_stride=256):
        super().__init__()
        self.signals = signals
        self.locations = locations
        self.durations = durations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride
        self.rng = np.random.default_rng()
        self.n_channels = signals[0].shape[0]
        self.rng = np.random.default_rng()
        key_array = []
        for i, array in enumerate(self.locations):
            n = (len(array) - self.window_size)//self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)
        x, location, duration = self.__data_generation__(keys)
        n_objects = np.float32(np.count_nonzero(duration)) # Approximate method of counting the # of events
        return x, location, duration, n_objects

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        stride = self.network_stride
        x = np.empty(shape=(self.batch_size, self.window_size*stride, 1), dtype=np.float32)
        location = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        duration = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            signal = self.signals[key[0]][key[1]*stride:stride*(key[1]+self.window_size)].T
            x[i, :, 0] = signal
            location[i, :, 0] = self.locations[key[0]][key[1]:key[1]+self.window_size]
            duration[i, :, 0] = self.durations[key[0]][key[1]:key[1]+self.window_size]
        return x, location, duration
    

def get_testing_data(data_path):
    CH = 18
    label_dict = {'eyem':0, 'chew':1, 'shiv':2, 'musc':3, 'elec':4}
    index = (label_dict['musc'], label_dict['chew'])
    signals, labels, _, _, _, test_idx = get_single_channel_indexed(
        data_path=data_path, CH=CH, index=index
    )
    signals = signals[test_idx]
    labels = labels[test_idx]
    return signals, labels


class EpochGenerator(tf.keras.utils.Sequence):
    def __init__(self, signals, labels, batch_size=128, shuffle=True, batch_stride=200, window_size=1024,
                 network_stride=256):
        super().__init__()
        self.signals = signals
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride
        self.rng = np.random.default_rng()
        self.n_channels = signals[0].shape[0]
        self.rng = np.random.default_rng()
        key_array = []
        for i, array in enumerate(self.labels):
            n = (len(array) - self.window_size)//self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)
        x, y = self.__data_generation__(keys)
        scaling = self.rng.integers(low=0, high=1, size=(self.batch_size, 1, 1), endpoint=True)
        scaling = 2. * scaling - 1.
        x = scaling * x
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        stride = self.network_stride
        x = np.empty(shape=(self.batch_size, self.window_size*stride, 1), dtype=np.float32)
        y = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        flip = self.rng.integers(low=0, high=1, size=self.batch_size, endpoint=True)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            signal = self.signals[key[0]][key[1]*stride:stride*(key[1]+self.window_size)].T
            y_ = self.labels[key[0]][key[1]:key[1]+self.window_size]
            if flip[i]:
                signal = np.flip(signal)
                y_ = np.flip(y_)
            x[i, :, 0] = signal
            y[i, :, 0] = y_
        return x, y


class NoAugEpochGenerator(tf.keras.utils.Sequence):
    def __init__(self, signals, labels, batch_size=128, shuffle=True, batch_stride=200, window_size=1024,
                 network_stride=256):
        super().__init__()
        self.signals = signals
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride
        self.rng = np.random.default_rng()
        self.n_channels = signals[0].shape[0]
        self.rng = np.random.default_rng()
        key_array = []
        for i, array in enumerate(self.labels):
            n = (len(array) - self.window_size)//self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size

    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        stride = self.network_stride
        x = np.empty(shape=(self.batch_size, self.window_size*stride, 1), dtype=np.float32)
        y = np.empty(shape=(self.batch_size, self.window_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            signal = self.signals[key[0]][key[1]*stride:stride*(key[1]+self.window_size)].T
            x[i, :, 0] = signal
            y[i, :, 0] = self.labels[key[0]][key[1]:key[1]+self.window_size]
        return x, y