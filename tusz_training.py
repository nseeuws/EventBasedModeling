import os
import tensorflow as tf
import numpy as np
import pandas as pd

import eventnet


def main(args):
    # Get path command line arguments
    data_path = args.data_path
    if args.network_path:
        network_path = args.network_path
    else:
        network_path = 'tuar_eventnet.h5'
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = 'tuar_eventnet_log.csv'

    # Sanity check - check for access to all the files
    assert os.path.isfile(data_path)
    assert os.access(network_path, mode=os.F_OK)
    assert os.access(log_path, mode=os.F_OK)

    # Get training command line arguments
    batch_size = args.batch_size
    learning_rate = args.lr
    n_epochs = args.n_epochs
    duration_factor = args.duration_factor

    # Get loss command line arguments
    lambda_r = args.lambda_r

    # Get network command line arguments
    duration_threshold = args.duration_threshold

    # Command line arguments sanity checks
    assert batch_size > 0
    assert learning_rate > 0.
    assert n_epochs > 0.
    assert lambda_r >= 0.  # You _can_ ignore duration loss if you want to
    assert duration_factor > 0
    assert duration_threshold > 0.

    # ========== ACTUAL SCRIPT ==========
    # Standard parameters (changing these will involve changing implementations as well
    fs = 200

    n_steps = 4
    stride_factor = 4
    stride = stride_factor ** n_steps
    down_factor = stride_factor ** 6
    input_duration = duration_factor * down_factor


    pass


if __name__ == '__main__':
    argument_parser = eventnet.utils.parser_builder()
    arguments = argument_parser.parse_args()
    main(args=arguments)
