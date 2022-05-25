import os

import eventnet


def main(args):
    # Get path command line arguments
    data_path = args.data_path
    if args.network_path:
        network_path = args.network_path
    else:
        network_path = 'tusz_eventnet.h5'
    if args.log_path:
        log_path = args.log_path
    else:
        log_path = 'tusz_eventnet_log.csv'

    # Sanity check - check for access to all the files
    assert os.path.isfile(data_path)
    with open(network_path, mode='a'):
        os.utime(network_path, None)  # Create empty file
    with open(log_path, mode='a'):
        os.utime(log_path, None)  # Create empty file

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

    # DATA
    # Load and prepare data
    training_targets, val_targets = eventnet.data.load_tusz_training_data(
        data_path=data_path, duration_threshold=duration_threshold,
        fs=fs, stride=stride, input_duration=input_duration
    )
    signals_train, centers_train, durations_train = training_targets
    signals_val, centers_val, durations_val = val_targets

    # Set up generators to produce training (and validation) batches
    generator = eventnet.training.TUSZGenerator(
        signals=signals_train, centers=centers_train, durations=durations_train,
        batch_size=batch_size, shuffle=True,
        window_size=input_duration // stride, batch_stride=input_duration // (2 * stride),
        network_stride=stride
    )
    val_generator = eventnet.training.TUSZGenerator(
        signals=signals_val, centers=centers_val, durations=durations_val,
        batch_size=batch_size, shuffle=False,
        window_size=input_duration // stride, batch_stride=input_duration // (2 * stride),
        network_stride=stride
    )

    # LEARNING SETUP
    # Network
    network = eventnet.network.build_seizure_eventnet(
        input_duration=input_duration
    )

    # TRAINING
    eventnet.training.training_loop(
        network=network,
        generator=generator, val_generator=val_generator,
        learning_rate=learning_rate, lambda_r=lambda_r,
        n_epochs=n_epochs, log_path=log_path, network_path=network_path
    )


if __name__ == '__main__':
    argument_parser = eventnet.utils.parser_builder()
    arguments = argument_parser.parse_args()
    main(args=arguments)
