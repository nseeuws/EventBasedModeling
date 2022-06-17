from typing import List, Tuple
import numpy as np
import scipy.optimize


ListOfEvents = List[Tuple[int, int]]  # Type hint for a list of events (start-stop tuple)


def iou_scoring(
        reference_list: ListOfEvents, hypothesis_list: ListOfEvents,
        iou_cutoff: float
) -> Tuple[int, int, int]:
    """
    Compute hits, misses, and false alarms under an IoU cutoff criterion
    :param reference_list: List of reference, ground-truth events
    :param hypothesis_list: List of hypothesis, predicted events
    :param iou_cutoff: IoU cutoff value to determine hits-misses. Float in the range (0, 1)
    :return: Tuple of integers, counts of hits, misses, and false alarms (in that order)
    """
    assert 0 <= iou_cutoff <= 1

    # Set up a linear assignment problem to find pairs of reference-hypothesis events
    cost = np.empty(shape=(len(reference_list), len(hypothesis_list)))

    for i, reference_event in enumerate(reference_list):
        for j, hypothesis_event in enumerate(hypothesis_list):
            cost[i, j] = compute_iou(reference_event, hypothesis_event)

    row_idx, col_idx = scipy.optimize.linear_sum_assignment(
        cost_matrix=cost, maximize=True
    )

    # Start counting hits, misses, and false alarms
    hits_reference = np.zeros(shape=(len(reference_list),), dtype=np.uint8)
    hits_hypothesis = np.zeros(shape=(len(hypothesis_list),), dtype=np.uint8)

    for reference_index, hypothesis_index in zip(row_idx, col_idx):
        if cost[reference_index, hypothesis_index] >= iou_cutoff:
            hits_reference[reference_index] = 1
            hits_hypothesis[hypothesis_index] = 1

    true_positive = int(np.sum(hits_hypothesis))
    false_negative = int(np.sum(hits_reference == 0))
    false_positive = int(np.sum(hits_hypothesis == 0))

    return true_positive, false_negative, false_positive


def ovlp_scoring(
        reference_list: ListOfEvents, hypothesis_list: ListOfEvents
) -> Tuple[int, int, int]:
    """
    Compute hits, misses, and false alarms under the "OVLP" criterion
    Based on the TUH Corpus evaluation code: https://isip.piconepress.com/projects/tuh_eeg/downloads/nedc_eval_eeg/
    :param reference_list: List of reference, ground-truth events
    :param hypothesis_list: List of hypothesis, predicted events
    :return: Tuple of integers, counts of hits, misses, and false alarms (in that order)
    """

    hit = 0  # Hit count
    miss = 0  # Miss count
    fa = 0  # False alarm count

    for event in reference_list:
        starts, stops = get_ovlp_events(
            start=event[0], stop=event[1], event_list=hypothesis_list
        )
        if len(starts) > 0:
            hit += 1
        else:
            miss += 1

    for event in hypothesis_list:
        starts, stops = get_ovlp_events(
            start=event[0], stop=event[1], event_list=reference_list
        )
        if len(starts) == 0:
            fa += 1

    return hit, miss, fa


def get_ovlp_events(
        start: int, stop: int, event_list: ListOfEvents
) -> Tuple[List[int], List[int]]:
    """
    Helper function for `ovlp_scoring`
    Returns the events in `event_list` that overlap with an event at `start, stop`
    :param start: Start index of the "test event"
    :param stop: Stop index of the "test event"
    :param event_list: List of events to test against
    :return: Lists of starts and stops in `event_list` that overlap with `(start, stop)`
    """
    list_of_starts = []
    list_of_stops = []

    for event in event_list:
        if (event[1] > start) and (event[0] < stop):
            list_of_starts.append(event[0])
            list_of_stops.append(event[1])

    return list_of_starts, list_of_stops


def compute_iou(event1: Tuple[int, int], event2: Tuple[int, int]) -> float:
    """
    Compute IoU between two events
    :param event1: First event
    :param event2: Second event
    :return: IoU value
    """
    start1, stop1 = event1
    start2, stop2 = event2

    intersection = max(0., min(stop1, stop2) - max(start1, start2))
    union = stop2 - start2 + stop1 - start1 - intersection

    return intersection / union
