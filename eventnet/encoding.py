from typing import List, Tuple
import numpy as np
from scipy import signal


def get_objects(label_array: np.ndarray) -> List[Tuple[int, int]]:
    # count_objects = 0
    objects = []
    edge_points = np.abs(np.diff(label_array))

    idx = np.nonzero(edge_points)[0]

    if label_array[0] == 0:
        for i in range(len(idx) // 2):
            objects.append((idx[2 * i], idx[2 * i + 1]))

        if len(idx) % 2 == 1:
            objects.append((idx[-1], len(edge_points)))
    else:
        if len(idx) == 0:
            objects.append((0, len(edge_points)))
        else:
            objects.append((0, idx[0]))

            for i in range((len(idx) - 1) // 2):
                objects.append((idx[2 * i + 1], idx[2 * (i + 1)]))

            if len(idx) % 2 == 0:
                objects.append((idx[-1], len(edge_points)))

    return objects


def get_kernel_maps(objects: List[Tuple[int, int]], duration: int):
    kernel_maps = []
    x_range = np.asarray(np.arange(duration), dtype=np.float32)

    for i_object in range(len(objects)):
        start_point, end_point = objects[i_object]
        center = (end_point + start_point) // 2
        scale = max(end_point - start_point, 1)
        alpha = 0.5
        sigma = alpha * scale / 6

        def object_kernel(x):
            return np.exp(-0.5 * np.square((x - center) / sigma))

        kernel_maps.append(object_kernel(x_range))
    return kernel_maps


def get_localization_map(labels_downsampled: np.ndarray) -> Tuple[np.ndarray, int]:
    objects = get_objects(label_array=labels_downsampled)

    # total_duration = len(labels_downsampled)
    total_duration = labels_downsampled.shape[0]

    target_maps = get_kernel_maps(objects=objects, duration=total_duration)

    if len(objects) > 0:
        target_map = np.maximum.reduce(target_maps)
    else:
        target_map = np.zeros(shape=labels_downsampled.shape, dtype=np.float32)

    return target_map, len(objects)


def get_regression_targets(labels: np.ndarray) -> List[Tuple[int, int]]:
    objects = get_objects(label_array=labels)
    return objects


def get_target_maps(
        labels: List[np.ndarray], stride: int,
        duration_cutoff: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Extract the two EventNet training targets from
    a list of labels (binary signal indicating foreground-background)
    :param labels: List of 1d binary arrays containing event annotations
    :param stride: Output stride, downsampling factor of EventNet
    compared to the input signal
    :param duration_cutoff: Maximum duration for EventNet predictions
    (expressed in number of time samples at the input sampling frequency)
    :return: Two lists of 1d arrays containing the center and
    duration training targets, in that order. The target signals
    are downsampled with a factor `stride`
    """
    centers = []
    durations = []

    for label in labels:
        label_down = label[::stride]
        center, n_objects = get_localization_map(labels_downsampled=label_down)
        duration = np.zeros(shape=center.shape, dtype=np.float32)
        if n_objects > 0:
            targets = get_regression_targets(labels=label)
            peaks, _ = signal.find_peaks(center, width=1, height=0.9)
            for i_peak, peak in enumerate(peaks):
                start = targets[i_peak][0]
                stop = targets[i_peak][1]
                duration[peak] = (stop - start) / duration_cutoff

        centers.append(center)
        durations.append(duration)
    return centers, durations
