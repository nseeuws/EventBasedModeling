from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.signal
import scipy.optimize
import scipy.ndimage

import torch
from torch.utils.data import DataLoader 
from torch import nn
import torchvision

import encoding


Event = Tuple[int, int]
PerformanceCount = Dict[
    str,
    Dict[str, int]
]
Performance = Dict[
    str,
    Dict[str, float]
]
PerformanceStorage = Dict[
    str,
    Dict[str, np.ndarray]
]


def validate_iou_list(iou_thresholds: List[float]) -> bool:
    return all(
        [0. <= threshold <= 1. for threshold in iou_thresholds]
    )

def get_iou(event1: Event, event2: Event) -> float:
    start1, stop1 = event1
    start2, stop2 = event2

    intersection = max(
        0., min(stop1, stop2) - max(start1, start2)
    )
    union = stop2 - start2 + stop1 - start1 - intersection + 1e-8

    return intersection / union

def get_recall(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray) -> np.ndarray:
    assert tp.shape == fn.shape
    assert fn.shape == fp.shape

    return tp / (tp + fn)

def get_precision(tp: np.ndarray, fn: np.ndarray, fp: np.ndarray) -> np.ndarray:
    assert tp.shape == fn.shape
    assert fn.shape == fp.shape

    return tp / (tp + fp)

def f1_score(precision: np.ndarray, recall: np.ndarray) -> float:
    f1_array = 2 * precision * recall / (precision + recall)
    f1 = np.nanmax(f1_array)
    
    return f1 # type: ignore

def average_precision_score(precision: np.ndarray, recall: np.ndarray) -> float:
    assert precision.shape == recall.shape

    return -np.nansum(np.diff(recall) * precision[:-1]) # type: ignore

def construct_performance_storage(
    iou_thresholds: List[float], confidence_thresholds: np.ndarray
) -> Tuple[PerformanceStorage, List[str]]:
    assert validate_iou_list(iou_thresholds)

    n_thresholds = len(confidence_thresholds)

    detection_performance = {}
    iou_key_list = [
        str(threshold) for threshold in iou_thresholds
    ]

    for key in iou_key_list:
        detection_performance[key] = {
            'hit': np.zeros(shape=(n_thresholds,), dtype=np.int32),
            'miss': np.zeros(shape=(n_thresholds,), dtype=np.int32),
            'fa': np.zeros(shape=(n_thresholds,), dtype=np.int32)
        }

    return detection_performance, iou_key_list

def get_reference_events(
    duration: np.ndarray, stride: int
) -> List[Event]:
    reference_centers = np.argwhere(duration)  # Should be 1D TODO: Add to documentation
    reference_events = []

    for center in reference_centers:
        start = int(stride * center - duration[center] / 2.)
        stop = int(stride * center + duration[center] / 2.)
        event = (start, stop)
        reference_events.append(event)

    return reference_events

def get_prediction_events(
    center_point: np.ndarray, duration: np.ndarray,
    stride: int, threshold: float, center: Optional[np.ndarray] = None, nms: bool = False
) -> List[Event]:
    peaks, _ = scipy.signal.find_peaks(center_point, width=1, height=threshold)

    prediction_events = []
    prediction_scores = []
    for peak in peaks:
        duration_prediction = duration[peak]

        start = int(stride * peak - duration_prediction / 2.)
        stop = int(stride * peak + duration_prediction / 2.)
        event = (start, stop)
        prediction_events.append(event)

        if nms:
            score_prediction = center[peak]  # type: ignore
            prediction_scores.append(score_prediction)
    
    if nms:
        n_events = len(prediction_events)
        boxes = np.zeros(shape=(n_events, 4), dtype=np.float32)
        confidence_values = np.zeros(shape=(n_events,), dtype=np.float32)
        for i, event in enumerate(prediction_events):
            start, stop = event
            boxes[i, 0] = float(start)
            boxes[i, 1] = 0.
            boxes[i, 2] = float(stop)
            boxes[i, 3] = 1.
            confidence_values[i] = prediction_scores[i]
        indices = torchvision.ops.nms(
            boxes=torch.from_numpy(boxes), scores=torch.from_numpy(confidence_values),
            iou_threshold=0.05
        ).numpy()

        placeholder = []
        for index in indices:
            placeholder.append(prediction_events[index])
        prediction_events = placeholder

    return prediction_events

def evaluate_event_matches(
    predicted_events: List[Event],
    reference_events: List[Event],
    iou_thresholds: List[float]
) -> PerformanceCount:
    """Count number of hits, misses, and false alarms

    Args:
        predicted_events (List[Event]): List of predicted events
        reference_events (List[Event]): List of reference events
        iou_thresholds (List[float]): List of IoU thresholds to consider

    Returns:
        Performance: Output. Just figure it out
    """
    assert validate_iou_list(iou_thresholds)

    n_predictions = len(predicted_events)
    n_references = len(reference_events)

    cost = np.zeros(shape=(n_references, n_predictions))

    for i, reference_event in enumerate(reference_events):
        for j, predicted_event in enumerate(predicted_events):
            cost[i, j] = get_iou(
                reference_event, predicted_event
            )

    row_indices, column_indices = scipy.optimize.linear_sum_assignment(
        cost_matrix=cost, maximize=True
    )

    # Bookkeeping for hits, misses, and false alarms
    hits_tracker = [
        {
            'ref': np.zeros(shape=(n_references), dtype=np.int32),
            'pred': np.zeros(shape=(n_predictions), dtype=np.int32)
        } for _ in iou_thresholds
    ]

    # Count hits, misses, ...
    for reference_index, predicted_index in zip(
        row_indices, column_indices
    ):
        cost_value = cost[reference_index, predicted_index]
        if cost_value:
            for threshold, tracker in zip(
                iou_thresholds, hits_tracker
            ):
                if cost_value >= threshold:
                    tracker['ref'][reference_index] = 1
                    tracker['pred'][predicted_index] = 1

    # Prep output
    detection_performance: PerformanceCount = {}
    key_list = [str(threshold) for threshold in iou_thresholds]
    for key, tracker in zip(key_list, hits_tracker):
        tp = np.sum(tracker['pred'], dtype=np.int32)
        fn = np.sum(tracker['ref'] == 0, dtype=np.int32)
        fp = np.sum(tracker['pred'] == 0, dtype=np.int32)

        detection_performance[key] = {
            'tp': tp,
            'fn': fn,
            'fp': fp
        }

    return detection_performance


@torch.no_grad()
def evaluate_event_detection(
    dataloader: DataLoader, model: nn.Module,
    iou_thresholds: List[float], 
    max_duration: int, stride: int,
    device: str = 'cuda',
) -> Performance:
    assert validate_iou_list(iou_thresholds)

    n_confidence_thresholds = 30
    thresholds = np.linspace(start=1e-3, stop=1, num=n_confidence_thresholds)

    performance, iou_key_list = construct_performance_storage(
        iou_thresholds=iou_thresholds, confidence_thresholds=thresholds
    ) 

    # Actual evaluation
    model.eval()
    for data in dataloader:
        signal = data[0]
        center = data[1]
        duration = data[2]
        network_output = model(signal.to(device))
        pred_loc = network_output[0]
        pred_size = network_output[1]
        pred_loc.cpu().numpy()
        pred_size.cpu().numpy()

        pred_size *= max_duration
        duration *= max_duration

        batch_size = signal.shape[0]
        for i_batch in range(batch_size):
            reference_list = get_reference_events(
                duration=duration[i_batch, 0, :].numpy(), stride=stride
            )

            for i_threshold, threshold in enumerate(thresholds):
                prediction_list = get_prediction_events(
                    center_point=pred_loc[i_batch, 0, :].cpu().numpy(),
                    duration=pred_size[i_batch, 0, :].cpu().numpy(),
                    stride=stride, threshold=threshold,
                    center=center[i_batch, 0, :].numpy(), nms=True
                )

                iou_count = evaluate_event_matches(
                    predicted_events=prediction_list,
                    reference_events=reference_list,
                    iou_thresholds=iou_thresholds
                )

                for key in iou_key_list:
                    tp = iou_count[key]['tp']
                    fn = iou_count[key]['fn']
                    fp = iou_count[key]['fp']

                    performance[key]['hit'][i_threshold] += tp
                    performance[key]['miss'][i_threshold] += fn
                    performance[key]['fa'][i_threshold] += fp

    # Processing results
    performance_result = {}
    for key in iou_key_list:
        keyed_results = performance[key]
        tp = keyed_results['hit']
        fn = keyed_results['miss']
        fp = keyed_results['fa']

        precision = get_precision(tp=tp, fn=fn, fp=fp)
        recall = get_recall(tp=tp, fn=fn, fp=fp)

        ap = average_precision_score(precision=precision, recall=recall)
        f1 = f1_score(precision=precision, recall=recall)

        performance_result[key] = {'ap': ap, 'f1': f1}

    return performance_result



def full_evaluate_event_matches(
    predicted_events: List[Event],
    reference_events: List[Event],
    iou_thresholds: List[float]
) -> PerformanceCount:
    """Count number of hits, misses, and false alarms

    Args:
        predicted_events (List[Event]): List of predicted events
        reference_events (List[Event]): List of reference events
        iou_thresholds (List[float]): List of IoU thresholds to consider

    Returns:
        Performance: Output. Just figure it out
    """
    assert validate_iou_list(iou_thresholds)

    n_predictions = len(predicted_events)
    n_references = len(reference_events)

    cost = np.zeros(shape=(n_references, n_predictions))


    for i, reference_event in enumerate(reference_events):
        for j, predicted_event in enumerate(predicted_events):
            cost[i, j] = get_iou(
                reference_event, predicted_event
            )

    row_indices, column_indices = scipy.optimize.linear_sum_assignment(
        cost_matrix=cost, maximize=True
    )

    # Bookkeeping for hits, misses, and false alarms
    hits_tracker = [
        {
            'ref': np.zeros(shape=(n_references), dtype=np.int32),
            'pred': np.zeros(shape=(n_predictions), dtype=np.int32),
            'durations': [],
            'centers': []
        } for _ in iou_thresholds
    ]

    # Count hits, misses, ...
    for reference_index, predicted_index in zip(
        row_indices, column_indices
    ):
        cost_value = cost[reference_index, predicted_index]
        if cost_value:
            for threshold, tracker in zip(
                iou_thresholds, hits_tracker
            ):
                if cost_value >= threshold:
                    tracker['ref'][reference_index] = 1
                    tracker['pred'][predicted_index] = 1
                    ref_start, ref_stop = reference_events[reference_index]
                    pred_start, pred_stop = predicted_events[predicted_index]

                    ref_duration = abs(ref_stop - ref_start)
                    pred_duration = abs(pred_stop - pred_start)
                    ref_center = (ref_start + ref_stop) / 2
                    pred_center = (pred_start + pred_stop) / 2

                    tracker['durations'].append(
                        float((pred_duration - ref_duration) / ref_duration)
                    )
                    tracker['centers'].append(
                        float((pred_center - ref_center) / ref_duration)
                    )

    # Prep output
    detection_performance: PerformanceCount = {}
    key_list = [str(threshold) for threshold in iou_thresholds]
    for key, tracker in zip(key_list, hits_tracker):
        tp = np.sum(tracker['pred'], dtype=np.int32)
        fn = np.sum(tracker['ref'] == 0, dtype=np.int32)
        fp = np.sum(tracker['pred'] == 0, dtype=np.int32)
        durations = tracker['durations']
        centers = tracker['centers']

        detection_performance[key] = {
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'durations': durations,
            'centers': centers
        }

    return detection_performance



@torch.no_grad()
def evaluate_epoch_detection(
    dataloader: DataLoader, unet: nn.Module,
    iou_thresholds: List[float], max_duration: int,
    post_processing: str,
    stride: int, device: str = 'cuda'
) -> Performance:
    assert validate_iou_list(iou_thresholds)

    n_confidence_thresholds = 30
    thresholds = np.linspace(start=1e-3, stop=1, num=n_confidence_thresholds)
    performance, iou_key_list = construct_performance_storage(
        iou_thresholds=iou_thresholds, confidence_thresholds=thresholds
    )

    event_container = []
    prediction_container = []

    def no_pp(y_thresh):
        return y_thresh
    def filter_pp(y_thresh):
        time_pp = 201 // stride + 1
        y_filt = scipy.signal.medfilt(
            volume=y_thresh, kernel_size=time_pp
        )
        return y_filt
    def morph_pp(y_thresh):
        min_duration = 205 // stride
        y_filt = scipy.ndimage.binary_closing(
            input=y_thresh,
            structure=np.ones(shape=(min_duration,), dtype=np.int8)
        )
        y_filt = scipy.ndimage.binary_opening(
            input=y_filt,
            structure=np.ones(shape=(min_duration,), dtype=np.int8)
        )
        return y_filt
    
    if post_processing == 'none':
        pp_function = no_pp
    elif post_processing == 'filter':
        pp_function = filter_pp
    elif post_processing == 'morphological':
        pp_function = morph_pp


    # Actual evaluation
    unet.eval()
    for data in dataloader:
        signal = data[0]
        duration = data[2]
        network_output = unet(signal.to(device))
        network_output = network_output.cpu().numpy()

        duration *= max_duration

        batch_size = signal.shape[0]
        for i_batch in range(batch_size):
            reference_list = get_reference_events(
                duration=duration[i_batch, 0, :].numpy(), stride=stride
            )
            event_container.append(reference_list)
            prediction_container.append(network_output[i_batch, 0, :])


            for i_threshold, threshold in enumerate(thresholds):
                y_thresh = np.asarray(
                    network_output[i_batch, 0, :] >= threshold,
                    dtype=np.int8
                )
                #y_filt = medfilt(
                #     volume=y_thresh, kernel_size=time_pp
                #) 
                #min_duration = 205 // stride
                #min_duration = time_pp
                #y_filt = scipy.ndimage.binary_closing(
                #    input=y_thresh,
                #    structure=np.ones(shape=(min_duration,), dtype=np.int8)
                #)
                #y_filt = scipy.ndimage.binary_opening(
                #    input=y_filt,
                #    structure=np.ones(shape=(min_duration,), dtype=np.int8)
                #)
                #y_filt = y_thresh
                # ---------------------------------------------------------------
                y_filt = pp_function(y_thresh=y_thresh) # type:ignore
                # ---------------------------------------------------------------

                hypotheses = encoding.get_objects(y_filt)
                prediction_list = [
                    (stride * event[0], stride * event[1]) for event in hypotheses
                ]

                iou_count = evaluate_event_matches(
                    predicted_events=prediction_list,
                    reference_events=reference_list,
                    iou_thresholds=iou_thresholds
                )

                for key in iou_key_list:
                    tp = iou_count[key]['tp']
                    fn = iou_count[key]['fn']
                    fp = iou_count[key]['fp']

                    performance[key]['hit'][i_threshold] += tp
                    performance[key]['miss'][i_threshold] += fn
                    performance[key]['fa'][i_threshold] += fp

    # Processing results
    performance_result = {}
    for key in iou_key_list:
        keyed_results = performance[key]
        tp = keyed_results['hit']
        fn = keyed_results['miss']
        fp = keyed_results['fa']

        precision = get_precision(tp=tp, fn=fn, fp=fp)
        recall = get_recall(tp=tp, fn=fn, fp=fp)

        ap = average_precision_score(precision=precision, recall=recall)

        performance_result[key] = {'ap': ap}
        f1 = 2 * precision * recall / (precision + recall)
        f1 = np.nanmax(f1)
        performance_result[key]['f1']  = f1

    return performance_result