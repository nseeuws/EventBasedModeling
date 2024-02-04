import h5py 
import numpy as np 
import tensorflow as tf 

import encoding

def load_eeg(path):
    with h5py.File(path, 'r') as f:
        file_names = []
        labels = []
        signals = []
        
        file_names_ds = f['filenames']
        signals_ds = f['signals']
        labels_ds = f['labels']
        
        for i in range(len(signals_ds)):
            file_names.append(file_names_ds[i])
            data = np.asarray(np.vstack(signals_ds[i]), dtype=np.float32).T
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            signals.append((data - mean) / std)
            labels.append(np.asarray(labels_ds[i], dtype=np.int8))
        
        return file_names, signals, labels
    
def purge_test_recordings(input_duration, files_training, signals_training, labels_training):
    files_purged = []
    signals_purged = []
    labels_purged = []
    for file, signal, label in zip(files_training, signals_training, labels_training):
        if len(label) > input_duration:
            files_purged.append(file)
            signals_purged.append(signal)
            labels_purged.append(label)
    return files_purged, signals_purged, labels_purged
    
def purge_recordings(input_duration, files_training, signals_training, labels_training, threshold):
    files_purged = []
    signals_purged = []
    labels_purged = []
    for file, signal, label in zip(files_training, signals_training, labels_training):
        if len(label) > input_duration:
            events = encoding.get_objects(label_array=label)
            if len(events) > 0:
                duration_flag = [(event[1] - event[0]) <= threshold for event in events]
                if all(duration_flag):
                    files_purged.append(file)
                    signals_purged.append(signal)
                    labels_purged.append(label)
            else:
                files_purged.append(file)
                signals_purged.append(signal)
                labels_purged.append(label)
    return files_purged, signals_purged, labels_purged

def find_seizure_recordings(labels):
    seizure_label = []
    for label in labels:
        seizure_label.append(np.sum(label) > 0)
    seizure_label = np.asarray(seizure_label, dtype=np.uint8)


class EventGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, signals, locations, durations, batch_size=32, shuffle=True,
        batch_stride=200, window_size=1024, network_stride=256
    ):
        super().__init__()
        self.signals = signals
        self.locations = locations
        self.durations = durations
        self.batch_size = batch_size
        self.stride = batch_stride
        self.shuffle = shuffle
        self.network_stride = network_stride
        self.window_size = window_size
        self.n_channels = signals[0].shape[1]

        key_array = []
        for i, array in enumerate(self.locations):
            n = (len(array) - self.window_size) // self.stride
            for j in range(n):
                key_array.append([i, self.stride*j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)
        self.on_epoch_end()

    def __len__(self):
        return len(self.key_array)//self.batch_size
        
    def __getitem__(self, index):
        keys = np.arange(start=index*self.batch_size, stop=(index+1)*self.batch_size)
        x, location, duration = self.__data_generation__(keys)
        n_objects = np.float32(np.count_nonzero(duration))
        return x, location, duration, n_objects

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        stride = self.network_stride
        x = np.empty(shape=(
            self.batch_size, self.window_size*stride, self.n_channels, 1
        ), dtype=np.float32)
        location = np.empty(shape=(self.batch_size, self.window_size, 1, 1), dtype=np.float32)
        duration = np.empty(shape=(self.batch_size, self.window_size, 1, 1), dtype=np.float32)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            x[i, :, :, 0] = self.signals[key[0]][key[1]*stride:stride*(key[1]+self.window_size), :]
            location[i, :, 0, 0] = self.locations[key[0]][key[1]:key[1]+self.window_size]
            duration[i, :, 0, 0] = self.durations[key[0]][key[1]:key[1]+self.window_size]
        return x, location, duration
    

class EpochGenerator(tf.keras.utils.Sequence):
    def __init__(
        self, signals, labels, batch_size=32, shuffle=True,
        batch_stride=200, window_size=1024
    ):
        super().__init__()
        self.signals = signals
        self.labels = labels
        self.batch_size = batch_size
        self.stride = batch_stride
        self.shuffle = shuffle
        self.window_size = window_size
        self.n_channels = signals[0].shape[1]

        key_array = []
        for i, array in enumerate(self.labels):
            n = (len(array) - self.window_size) // self.stride
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
        x = np.empty(shape=(
            self.batch_size, self.window_size, self.n_channels, 1
        ), dtype=np.float32)
        y = np.empty(shape=(self.batch_size, self.window_size, 1, 1), dtype=np.float32)
        for i in range(self.batch_size):
            key = self.key_array[keys[i]]
            x[i, :, :, 0] = self.signals[key[0]][key[1]:(key[1]+self.window_size), :]
            y[i, :, 0, 0] = self.labels[key[0]][key[1]:key[1]+self.window_size]
        return x, y