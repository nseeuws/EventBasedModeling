import numpy as np 
import scipy.optimize 

import utils

def get_recall(tp, fn, fp):
    return tp / (tp + fn)


def get_precision(tp, fn, fp):
    return tp / (tp + fp)

def compute_f1_score(prec_array, rec_array):
    prec_array = np.asarray(prec_array)
    rec_array = np.asarray(rec_array)
    
    return 2 * (prec_array*rec_array) / (prec_array + rec_array)

def performance_evaluation(
    ref_obj: list, hyp_obj: list, iou_threshold: float
):
    """Evaluate detection performance for given reference and 
    predicted events.

    Args:
        ref_obj (list): List of reference event tuples (start, stop)
        hyp_obj (list): List of hypothesis event tuples (start, stop)
        iou_thresholds (float): Threshold between 0 and 1, indicating 
        the IoU level considered a "clean hit"

    Returns:
        list: List of hits, misses, and FA at the given IoU threshold,
        and duration true and predicted values at IoU threshold and >0 IoU.
    """
    cost = np.empty(shape=(len(ref_obj), len(hyp_obj)))

    for i, ref_event in enumerate(ref_obj):
        for j, hyp_event in enumerate(hyp_obj):
            cost[i, j] = utils.get_iou(ref_event, hyp_event)
    
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(
        cost_matrix=cost, maximize=True
    )

    # Hits, misses, and false alarms
    hits_ref = np.zeros(shape=(len(ref_obj),), dtype=np.uint8) # Counting hits or misses in the ref events
    hits_hyp = np.zeros(shape=(len(hyp_obj),), dtype=np.uint8) # Counting hits or misses in the hyp events


    for ref_index, hyp_index in zip(row_ind, col_ind):
        if cost[ref_index, hyp_index] >= iou_threshold:
            hits_ref[ref_index] = 1
            hits_hyp[hyp_index] = 1
    tp = np.sum(hits_hyp)
    fn = np.sum(hits_ref == 0)
    fp = np.sum(hits_hyp == 0)
              
    return (tp, fn, fp)

def ovlp_score(ref_a, hyp_a):

        # prime the output strings with null characters
        #

        # loop over the ref annotation to collect hits and misses
        #
    hit = int(0)
    mis = int(0)
    fal = int(0)

    for event in ref_a:         
        starts, stops = get_events(event[0], event[1], hyp_a)
        if len(starts)!=0:
            hit += 1
        else:
            mis +=1

        # loop over the hyp annotation to collect false alarms
        #
    for event in hyp_a:
        starts, stops = get_events(event[0], event[1], ref_a)
        if len(starts)==0:
            fal += 1

        # exit gracefully
        #
    return hit, mis, fal


def get_events(start_a, stop_a, events_a):

        # declare output variables
        #
    #labels = []
    starts = []
    stops = []

        # loop over all events
        #
    for event in events_a:

            # if the event overlaps partially with the interval,
            # it is a match. this means:
            #              start               stop
            #   |------------|<---------------->|-------------|
            #          |---------- event -----|
            #
        if (event[1] > start_a) and (event[0] < stop_a):
            starts.append(event[0])
            stops.append(event[1])
            #labels.append(event[2])
        
        # exit gracefully
        #
    return [starts, stops]