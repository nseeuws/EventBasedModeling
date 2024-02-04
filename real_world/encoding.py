import numpy as np 
import scipy.signal
import tensorflow as tf


def decode_artefact_events(loc, size, thresh, stride):
    """Decode artefact events

    Args:
        loc (list): Centerpoint predictions
        size (list): Duration predictions
        thresh (float): Cut-off threshold for peak detection
        stride (float): Network output stride

    Returns:
        list: List of decoded events
    """
    n_avg = 5
    peaks, _ = scipy.signal.find_peaks(loc[0, :, 0], width=1, height=thresh)

    hyp_conf = []
    hyp_obj = []

    for peak in peaks:
        duration_prediction = np.nanmean(size[0, peak, 0])
        #duration_prediction = size[0, peak, 0]
        start = int(stride * peak - duration_prediction/2.)
        stop = int(stride * peak + duration_prediction/2.)
        hyp_conf.append(loc[0, peak, 0])
        hyp_obj.append([start, stop])

    if len(hyp_obj)>0:
            hyp_tensor = np.zeros(shape=(len(hyp_obj), 4), dtype=np.float32)
            hyp_tensor[:, 1] = 0
            hyp_tensor[:, 3] = 1
       
            for i in range(len(hyp_obj)):
                hyp_tensor[i, 0] = hyp_obj[i][0]
                hyp_tensor[i, 2] = hyp_obj[i][1]
                idx = tf.image.non_max_suppression(boxes=hyp_tensor, scores=hyp_conf,
                                            max_output_size=len(hyp_tensor), iou_threshold=0.5)
            hyp_obj = np.asarray(hyp_obj)[idx, :]  # type: ignore
    return hyp_obj 


def get_objects(label_array):
    """Extract events from a background-foreground time signal.

    Args:
        label_array (np.ndarray): 1D time signal array with zero values for
        background samples and positive values for foreground samples.

    Returns:
        list: List of (start, stop) tuples
    """
    #count_objects = 0
    objects = []
    edge_points = np.abs(np.diff(label_array))
    
    idx = np.nonzero(edge_points)[0]
    
    if label_array[0] == 0:
        for i in range(len(idx)//2):
            objects.append([idx[2*i], idx[2*i+1]])
        
        if len(idx)%2 == 1:
            objects.append([idx[-1], len(edge_points)])
    else:
        if len(idx) == 0:
            objects.append([0, len(edge_points)])
        else:
            objects.append([0, idx[0]])
    
            for i in range((len(idx)-1)//2):
                objects.append([idx[2*i+1], idx[2*(i+1)]])
    
            if len(idx)%2 ==0:
                objects.append([idx[-1], len(edge_points)])
            
    return objects


def get_kernel_maps(objects, duration):
    """Create a "kernel map" list, each of size duration for every object.

    Args:
        objects (list): List of (start, stop) object tuples
        duration (int): Duration of the created map, in number of samples

    Returns:
        list: List of duration-sized arrays, one array for every object.
    """
    kernel_maps = []
    x_range = np.asarray(np.arange(duration), dtype=np.float32)

    for i_object in range(len(objects)):
        start_point, end_point = objects[i_object]
        center = (end_point + start_point) // 2
        scale = max(end_point - start_point, 1)
        alpha = 0.5
        sigma = alpha * scale/6
    
        def object_kernel(x):
            return np.exp(-0.5*np.square((x - center)/sigma))
    
        kernel_maps.append(object_kernel(x_range))
    return kernel_maps


def get_localization_map(labels_downsampled):
    """Create a target array for center point prediction
    based on the focal loss.

    Args:
        labels_downsampled (np.ndarray): 1D-array with time samples
        indicating foreground vs. background

    Returns:
        np.ndarray: 1D target array, after kernel smoothing
        int: Number of objects found in the input array
    """
    objects = get_objects(label_array=labels_downsampled)
    
    #total_duration = len(labels_downsampled)
    total_duration = labels_downsampled.shape[0]
    
    target_maps = get_kernel_maps(objects=objects, duration=total_duration)
    
    if len(objects)>0:
        target_map = np.maximum.reduce(target_maps)
    else:
        target_map = np.zeros(shape=labels_downsampled.shape, dtype=np.float32)
    
    return target_map, len(objects)


def get_regression_targets(labels):
    objects = get_objects(label_array=labels)
    return objects


def get_target_maps(labels, stride=16, log=True):
    """Generate "target maps" from the labels array. Directly generate
    the learning targets for my duration and location head.

    Args:
        labels (list): List of label arrays
        stride (int): Stride of the output targets, by how much do you downsample?
        log (bool): Return log durations or not

    Returns:
        tuple: Tuple with `location` and `duration` targets
    """
    locations = []
    durations = []
    for label in labels:
        label_down = label[::stride]
        location, n_objects = get_localization_map(labels_downsampled=label_down)
        duration = np.zeros(shape=location.shape, dtype=np.float32)
        if n_objects > 0:
            targets = get_regression_targets(labels=label)
            peaks, _ = scipy.signal.find_peaks(location, width=1, height=0.9)
            for i_peak, peak in enumerate(peaks):
                start = targets[i_peak][0]
                stop = targets[i_peak][1]
                if log:
                    duration[peak] = np.log(max(stop - start, 1e-8))
                else:
                    duration[peak] = stop - start
                    
        locations.append(location)
        durations.append(duration)
    #return np.asarray(locations), np.asarray(durations)
    return locations, durations