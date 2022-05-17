import numpy as np
from scipy import signal
import tensorflow as tf
from typing import List, Tuple


def decode_eventnet_outputs(
        center: np.ndarray, duration: np.ndarray,
        threshold: float, stride: int,
        duration_cutoff: int
) -> List[Tuple[int, int]]:
    """
    Decode the two output signals of an EventNet into
    a list of events.
    :param center: 1d(!) array of the center signal predictions
    :param duration: 1d(!) array of the duration predictions
    :param threshold: Confidence cutoff for the center predictions
    :param stride: Output stride, downsampling factor of EventNet
    compared to the input signal
    :param duration_cutoff: Maximum duration for EventNet predictions
    (expressed in number of time samples at the input sampling frequency)
    :return: A list of events (expressed as start and stop indices at
    the original sampling frequencies)
    """
    peaks, _ = signal.find_peaks(
        x=center, width=1, height=threshold
    )

    prediction_conf = []
    prediction_obj = []

    for peak in peaks:
        duration_prediction = duration[peak]
        duration_prediction *= duration_cutoff

        start = stride * peak - duration_prediction // 2
        stop = stride * peak + duration_prediction // 2

        prediction_conf.append(center[peak])
        prediction_obj.append((start, stop))

    if len(prediction_obj) > 0:
        prediction_tensor = np.zeros(shape=(len(prediction_obj), 4), dtype=np.float32)
        prediction_tensor[:, 1] = 0
        prediction_tensor[:, 3] = 1

        for i in range(len(prediction_obj)):
            prediction_tensor[i, 0] = prediction_obj[i][0]
            prediction_tensor[i, 2] = prediction_obj[i][1]
        idx = tf.image.non_max_suppression(
            boxes=prediction_tensor, scores=prediction_conf,
            max_output_size=len(prediction_tensor), iou_threshold=0.5
        )
        prediction_obj = np.asarray(prediction_obj)[idx, :]
    return prediction_obj
