from typing import Tuple, List
import numpy as np
import h5py
from sklearn import model_selection

import eventnet.encoding
import utils
import encoding

# Typing aides
data_container = List[np.ndarray]
target_container = Tuple[
    data_container, data_container, data_container
]


def _load_tuar_data_from_disk(data_path: str)\
        -> Tuple[
            List[str], List[np.ndarray], List[np.ndarray], List[np.ndarray]
        ]:
    """
    Load TUAR data from disk
    :param data_path: Path to the HDF5 object
    :return: List of file names, signals, channel-level labels, and
    cross-channel labels, in that order.
    """
    with h5py.File(data_path, 'r') as file:
        file_names = []
        signals = []
        channel_labels = []
        labels = []

        file_names_ds = file['filenames']
        signals_ds = file['signals']
        channel_labels_ds = file['ch_labels']
        labels_ds = file['labels']

        for i in range(len(file_names_ds)):
            file_names.append(file_names_ds[i])
            data = np.asarray(np.vstack(signals_ds[i]), dtype=np.float32)
            mean = np.mean(data, axis=-1, keepdims=True)
            std = np.std(data, axis=-1, keepdims=True)
            signals.append((data - mean) / std)
            channel_labels.append(channel_labels_ds[i])
            labels.append(np.stack(
                [np.vstack(channel_array) for channel_array in labels_ds[i]],
                axis=0))

        return file_names, signals, channel_labels, labels


def _stack_tuar_data(
        signals: List[np.ndarray], labels: List[np.ndarray],
        file_names: List[str], n_channels: int, index: Tuple[int, ...]
) -> Tuple[
    List[np.ndarray], List[np.ndarray], List[str]
]:
    """
    Stack multi-channel data into a list of single-channel data,
    based on the given index (merging event labels for classes in the index)
    :param signals: List of signal data
    :param labels: List of label data
    :param file_names: List of file names
    :param n_channels: Number of channels in the multi-channel data
    :param index: Index for event labels
    :return: Stacked signals, labels, and file names, in that order
    """
    signals_stacked = []
    labels_stacked = []
    names_stacked = []

    for i in range(len(signals)):
        for i_ch in range(n_channels):
            signals_stacked.append(signals[i][i_ch])
            if len(index) == 1:
                labels_stacked.append(labels[i][i_ch, index[0], :])
            else:
                labels_stacked.append(np.max(labels[i][i_ch, index, :], axis=0))
            names_stacked.append(file_names[i][0:8])

    return signals_stacked, labels_stacked, names_stacked


def _get_indexed_tuar_data(
        data_path: str, n_channels: int, index: Tuple[int, ...]
):
    file_names, signals, _, labels = _load_tuar_data_from_disk(data_path)

    signals_stacked, labels_stacked, names_stacked = _stack_tuar_data(
        signals=signals, labels=labels, file_names=file_names,
        n_channels=n_channels, index=index
    )

    # Find signals containing targets
    targets = [np.sum(labels) for labels in labels_stacked]

    # Split data into training and validation
    fold_splitter = utils.stratified_group_k_fold(
        X=signals_stacked, y=targets, groups=names_stacked,
        k=5, seed=1234
    )
    signals_stacked = np.asarray(signals_stacked, dtype=object)
    labels_stacked = np.asarray(labels_stacked, dtype=object)
    names_stacked = np.asarray(names_stacked)
    targets = np.asarray(targets)
    idx, test_idx = next(fold_splitter)
    training_fold_splitter = utils.stratified_group_k_fold(
        X=signals_stacked[idx], y=targets[idx], groups=names_stacked[idx],
        k=5, seed=42
    )
    train_idx, val_idx = next(training_fold_splitter)

    return signals_stacked, labels_stacked, idx, train_idx, val_idx, test_idx


def _get_tuar_training_data(data_path: str)\
        -> Tuple[
            data_container, data_container, data_container, data_container
        ]:
    n_channels = 18
    label_dict = {'eyem': 0, 'chew': 1, 'shiv': 2, 'musc': 3, 'elec': 4}
    index = (label_dict['musc'], label_dict['chew'])  # We're only looking at muscle and chewing

    # Load the data
    signals, labels, indices, train_indices, val_indices, _ = _get_indexed_tuar_data(
        data_path=data_path, n_channels=n_channels,
        index=index
    )

    # Extract training and validation data
    signals_train = signals[indices][train_indices]
    signals_val = signals[indices][val_indices]
    labels_train = labels[indices][train_indices]
    labels_val = labels[indices][val_indices]

    return signals_train, labels_train, signals_val, labels_val


def _filter_tuar_data(
        signals: List[np.ndarray], labels: List[np.ndarray],
        duration_threshold: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    signals_filtered = []
    labels_filtered = []

    for signal, label in zip(signals, labels):
        events = encoding.get_objects(label_array=label)

        if len(events) > 0:
            duration = [(event[1] - event[0]) <= duration_threshold
                        for event in events]
            if all(duration):
                signals_filtered.append(signal)
                labels_filtered.append(label)
    return signals_filtered, labels_filtered


def _filter_tusz_data(
        signals: List[np.ndarray], labels: List[np.ndarray],
        duration_threshold: int, input_duration: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    signals_filtered = []
    labels_filtered = []

    for signal, label in zip(signals, labels):
        if len(label) > input_duration:  # Is the recording long enough?
            events = eventnet.encoding.get_objects(label)
            if len(events) > 0:
                # If there are events, are they not too long?
                duration_flag = [(event[1] - event[0]) <= duration_threshold
                                 for event in events]
                if all(duration_flag):
                    signals_filtered.append(signal)
                    labels_filtered.append(label)
            else:  # No events => added to the list
                signals_filtered.append(signal)
                labels_filtered.append(label)
    return signals_filtered, labels_filtered


def _get_tusz_training_data(data_path: str)\
        -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    with h5py.File(data_path) as file:
        file_names = []
        labels = []
        signals = []

        file_names_ds = file['filenames']
        signals_ds = file['signals']
        labels_ds = file['labels']

        for i in range(len(signals_ds)):
            file_names.append(file_names_ds[i])
            data = np.asarray(np.vstack(signals_ds[i]), dtype=np.float32).T
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            signals.append((data - mean) / std)
            labels.append(np.asarray(labels_ds[i], dtype=np.int8))

    return file_names, signals, labels


def _split_tusz_data(signals: data_container, labels: data_container)\
        -> Tuple[data_container, data_container, data_container, data_container]:
    # Find out what recordings contain seizures
    recording_labels = [
        np.sum(label) > 0 for label in labels
    ]

    # Stratify splits based on seizure content
    signals_train, signals_val, labels_train, labels_val = model_selection.train_test_split(
        signals, labels, test_size=0.2, random_state=42, stratify=recording_labels
    )

    return signals_train, labels_train, signals_val, labels_val


def load_tuar_training_data(
        data_path: str, duration_threshold: float,
        stride: int, fs=200
) -> Tuple[target_container, target_container]:
    scaled_duration = int(duration_threshold * fs)
    # Load data
    signals_train, labels_train, signals_val, labels_val = _get_tuar_training_data(
        data_path=data_path
    )

    # Purge long events
    signals_train, labels_train = _filter_tuar_data(
        signals=signals_train, labels=labels_train,
        duration_threshold=scaled_duration
    )
    signals_val, labels_val = _filter_tuar_data(
        signals=signals_val, labels=labels_val,
        duration_threshold=scaled_duration
    )

    # Construct "target maps"
    centers_train, durations_train = encoding.get_target_maps(
        labels=labels_train, stride=stride, duration_cutoff=scaled_duration
    )
    centers_val, durations_val = encoding.get_target_maps(
        labels=labels_val, stride=stride, duration_cutoff=scaled_duration
    )

    # Create output variables
    training_targets = (
        signals_train, centers_train, durations_train
    )
    val_targets = (
        signals_val, centers_val, durations_val
    )

    return training_targets, val_targets


def load_tusz_training_data(
        data_path: str, duration_threshold: float,
        stride: int, fs: int, input_duration: int
) -> Tuple[target_container, target_container]:
    scaled_threshold = int(duration_threshold * fs)

    # Load data from disk
    _, signals, labels = _get_tusz_training_data(
        data_path=data_path
    )

    # Purge recordings that are too short, and contain events that are too long
    # (only for training)
    signals, labels = _filter_tusz_data(
        signals=signals, labels=labels,
        duration_threshold=scaled_threshold, input_duration=input_duration
    )

    # Split data into training and validation
    signals_train, labels_train, signals_val, labels_val = _split_tusz_data(
        signals=signals, labels=labels
    )

    # Prepare learning targets
    centers_train, durations_train = encoding.get_target_maps(
        labels=labels_train, stride=stride, duration_cutoff=scaled_threshold
    )
    centers_val, durations_val = encoding.get_target_maps(
        labels=labels_val, stride=stride, duration_cutoff=scaled_threshold
    )

    # Package outputs
    training_targets = (
        signals_train, centers_train, durations_train
    )
    val_targets = (
        signals_val, centers_val, durations_val
    )

    return training_targets, val_targets
