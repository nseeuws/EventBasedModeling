import logging 
import os 


def get_iou(event1, event2):
    start1, stop1 = event1
    start2, stop2 = event2
    
    intersection = max(0., min(stop1, stop2) - max(start1, start2)) 
    union = stop2 - start2 + stop1 - start1 - intersection
    
    return intersection/union


def set_tf_loglevel(level):
    if level >= logging.FATAL:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if level >= logging.ERROR:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    if level >= logging.WARNING:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    logging.getLogger('tensorflow').setLevel(level)