from typing import Tuple, List

import argparse
import logging

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch import nn 


import data
import utils
import network
import losses
import evaluation


def f1_score(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    f1_array = 2 * precision * recall / (precision + recall)
    return f1_array


def f1_index(precision: np.ndarray, recall: np.ndarray) -> int:
    f1_array = 2 * precision * recall / (precision + recall)
    index = np.nanargmax(f1_array)
    return int(index)


@torch.no_grad()
def evaluate_with_validation_threshold(
    dataloader: DataLoader, model: nn.Module, dataloader_val: DataLoader,
    iou_thresholds: List[float], 
    max_duration: int, stride: int,
    device: str = 'cuda',
) -> evaluation.Performance:
    assert evaluation.validate_iou_list(iou_thresholds)

    n_confidence_thresholds = 50
    thresholds = np.linspace(start=1e-5, stop=1, num=n_confidence_thresholds)


    # F1 threshold calculation
    performance, iou_key_list = evaluation.construct_performance_storage(
        iou_thresholds=iou_thresholds, confidence_thresholds=thresholds
    ) 
    f1_threshold_index = {}
    for key in iou_key_list:
        f1_threshold_index[key] = 0
    model.eval()

    for data in dataloader_val:
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
            reference_list = evaluation.get_reference_events(
                duration=duration[i_batch, 0, :].numpy(), stride=stride
            )

            for i_threshold, threshold in enumerate(thresholds):
                prediction_list = evaluation.get_prediction_events(
                    center_point=pred_loc[i_batch, 0, :].cpu().numpy(),
                    duration=pred_size[i_batch, 0, :].cpu().numpy(),
                    stride=stride, threshold=threshold,
                    center=center[i_batch, 0, :].numpy(), nms=True
                )

                iou_count = evaluation.full_evaluate_event_matches(
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

        precision = evaluation.get_precision(tp=tp, fn=fn, fp=fp)
        recall = evaluation.get_recall(tp=tp, fn=fn, fp=fp)

        index = f1_index(precision=precision, recall=recall)
        f1_threshold_index[key] = index
    

    # Actual evaluation
    performance, iou_key_list = evaluation.construct_performance_storage(
        iou_thresholds=iou_thresholds, confidence_thresholds=thresholds
    ) 
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
            reference_list = evaluation.get_reference_events(
                duration=duration[i_batch, 0, :].numpy(), stride=stride
            )

            for i_threshold, threshold in enumerate(thresholds):
                prediction_list = evaluation.get_prediction_events(
                    center_point=pred_loc[i_batch, 0, :].cpu().numpy(),
                    duration=pred_size[i_batch, 0, :].cpu().numpy(),
                    stride=stride, threshold=threshold,
                    center=center[i_batch, 0, :].numpy(), nms=True
                )

                iou_count = evaluation.full_evaluate_event_matches(
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

        precision = evaluation.get_precision(tp=tp, fn=fn, fp=fp)
        recall = evaluation.get_recall(tp=tp, fn=fn, fp=fp)

        ap = evaluation.average_precision_score(precision=precision, recall=recall)
        f1_array = f1_score(precision=precision, recall=recall)
        f1 = f1_array[f1_threshold_index[key]]

        performance_result[key] = {'ap': ap, 'f1': f1, 'precision': precision, 'recall': recall}

    return performance_result


@torch.no_grad()
def run_test_performance(
    model: nn.Module, ds: data.FiniteDataset,
    ds_val: data.FiniteDataset,
    iou_thresholds: List[float],
    batch_size: int = 128,
    network_stride: int = 4**2, max_duration: float = 0.3,
    input_duration: int = 4 * (4**5)
):
    model.eval()
    dl = DataLoader(
        dataset=ds, batch_size=batch_size,
        num_workers=6
    )
    dl_val = DataLoader(
        dataset=ds_val, batch_size=batch_size,
        num_workers=6
    )
    result = evaluate_with_validation_threshold(
        dataloader=dl, model=model, dataloader_val=dl_val,
        iou_thresholds=iou_thresholds, stride=network_stride,
        max_duration=int(max_duration * input_duration)
    )
    return result


def run_training(
        logger, n_epochs: int, 
        ds_train: data.FiniteDataset,
        iou_thresholds: List[float],
        learning_rate: float, batch_size: int,
        network_stride: int, max_duration: float,
        input_duration: int, evaluation_frequency: int
):
    device = 'cuda'
    one = torch.tensor(1., dtype=torch.float32, device=device)
    lambda_r = torch.tensor(5., dtype=torch.float32, device=device)

    # Data sets
    dl_train = DataLoader(
        dataset=ds_train, batch_size=batch_size,
        shuffle=True, num_workers=6
    )

    # Model
    model = network.Event()
    model.to(device)
    logger.info(f'Number of parameters: {utils.count_parameters(model)}')

    # Losses
    focal_loss = losses.FocalLoss(
        alpha=2., beta=4., a_t=0.1, device=device
    )
    regression_loss = losses.iou_loss

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.5)

    def train_loop(
        dataloader: DataLoader, model: nn.Module, 
        optimizer: torch.optim.Optimizer
    ):
        size = len(dataloader)

        total_loss = torch.tensor(
            0., dtype=torch.float32, device=device, requires_grad=False
        )
        f_total_loss = torch.tensor(0., dtype=torch.float32, device=device)
        r_total_loss = torch.tensor(0., dtype=torch.float32, device=device)
        gradient_norms = torch.zeros(size=(size,), device=device)
        model.train()

        for signal, center, duration, label in dataloader:
            n_objects = torch.count_nonzero(duration)

            # Compute the loss
            pred_loc, pred_size, pred_logit = model(signal.to(device))
            f_loss = focal_loss(
                center_target=center.to(device), center_pred=pred_loc,
                logit_pred=pred_logit
            ) / torch.maximum(one, n_objects)
            f_total_loss += f_loss

            if n_objects > 0:
                r_loss = regression_loss(
                    dur_target=duration.to(device), dur_pred=pred_size
                ) / torch.maximum(one, n_objects)
                r_total_loss += r_loss
                loss = f_loss + lambda_r * r_loss
            else:
                loss = f_loss

            total_loss += loss
            # Backpropagate
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.) # type: ignore
            optimizer.step()

        total_loss = total_loss.detach().cpu().numpy() / size
        f_total_loss = f_total_loss.detach().cpu().numpy() / size
        r_total_loss = r_total_loss.detach().cpu().numpy() / size
        gradient_norms = gradient_norms.cpu().numpy()
        
        return total_loss, f_total_loss, r_total_loss
    
    for epoch in range(n_epochs):
        logger.info(f'==== Epoch #{epoch:3d} ====')
        loss, f_loss, r_loss = train_loop(
            dataloader=dl_train, model=model, optimizer=optimizer
        )
        scheduler.step()

        logger.info(f'Loss      - {loss:0.3f}')
        logger.info(f'Center    - {f_loss:0.3f}')
        logger.info(f'Duration  - {r_loss:0.3f}')

        if (epoch + 1) % evaluation_frequency == 0:
            logger.info('===============================')
            logger.info('==== Detection performance ====')
            logger.info('===============================')

            logger.info('---------- Training -----------')
            detection_performance = evaluation.evaluate_event_detection(
                dataloader=dl_train, model=model,
                iou_thresholds=iou_thresholds, stride=network_stride,
                max_duration=int(max_duration * input_duration)
            )
            for key in detection_performance.keys():
                ap = detection_performance[key]['ap']
                f1 = detection_performance[key]['f1']
                logger.info(f'IoU - {key}')
                logger.info(f'AP:    {ap:0.3f}')
                logger.info(f'F1:    {f1:0.3f}')
    return model


def main(args):
    n_iterations = args.n_runs
    n_epochs = 50
    evaluation_frequency = 25
    learning_rate = 5e-4

    formatter = logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S' 
    )
    directory = args.result

    logger = utils.setup_logger(
        name='Root logger', log_file=f'{directory}/logger.log',
        formatter=formatter, level=logging.INFO
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f'Using {device}')

    logger.info('----------- Experiment ------------')
    event_proportion = args.event_proportion
    assert 0. <= event_proportion <= 1.
    logger.info(f'Event proportion:     {event_proportion}')
    logger.info(f'Number of epochs:     {n_epochs}')
    logger.info(f'Learning rate:        {learning_rate}')

    n_steps = 5
    stride_factor = 4
    network_stride = stride_factor**2
    stride = stride_factor**n_steps
    duration_factor = 4
    input_duration = duration_factor * stride
    max_duration = 0.3
    
    iou_thresholds = [0.25, 0.75]

    # Data
    if args.training_data is not None:
        signals_train, signals_val, signals_test = data.load_ecg_signals(
            training_path=args.training_data, test_path=args.test_data
        )
        noise_examples = data.load_ecg_noise(
            noise_type='em', path=args.noise_data
        )
    else:
        signals_train, signals_val, signals_test = data.load_ecg_signals()
        noise_examples = data.load_ecg_noise(noise_type='em')


    train_ds = data.DataGenerator(
        ecg_examples=signals_train, 
        noise_examples=noise_examples[0:1], # type: ignore
        batch_stride=input_duration // 4, window_size=input_duration,
        network_stride=network_stride, max_n_events=2,
        max_duration=max_duration,
        event_proportion=event_proportion
    )

    val_ds = data.DataGenerator(
        ecg_examples=signals_val,
        noise_examples=noise_examples[0:1], # type: ignore
        batch_stride=input_duration // 4, window_size=input_duration,
        network_stride=network_stride, max_n_events=2,
        max_duration=max_duration,
        event_proportion=event_proportion
    )
    test_ds = data.DataGenerator(
        ecg_examples=signals_test, noise_examples=noise_examples[1:2],
        batch_stride=input_duration // 4, window_size=input_duration,
        network_stride=network_stride, max_n_events=2,
        max_duration=max_duration,
        event_proportion=event_proportion
    )

    logger.info('=================')
    logger.info('==== RUNNING ====')
    logger.info('=================')
    
    test_performance_tracker = []

    for iteration in range(n_iterations):
        logger.info(f'---- Iteration {iteration} ----')
        iteration_logger = utils.setup_logger(
            name=f'Logging iteration {iteration}',
            log_file=f'{directory}/iter_{iteration}.log',
            formatter=formatter, level=logging.INFO
        ) 
        finite_train_ds, finite_val_ds, finite_test_ds = data.build_finite_datasets(
            train_ds=train_ds, val_ds=val_ds,
            test_ds=test_ds, input_duration=input_duration,
            network_stride=network_stride
        )
        training_output = run_training( # type: ignore
            logger=iteration_logger,
            n_epochs=n_epochs, ds_train=finite_train_ds,
            iou_thresholds=iou_thresholds,
            evaluation_frequency=evaluation_frequency,
            learning_rate=learning_rate
        )
        logger.info('-------------------')
        performance_result = run_test_performance(
            model=training_output, ds=finite_test_ds, # type: ignore
            ds_val=finite_val_ds,
            iou_thresholds=iou_thresholds
        )
        for iou in iou_thresholds:
            logger.info(f'# {iou:0.2f} IoU')
            ap = performance_result[str(iou)]['ap']
            f1 = performance_result[str(iou)]['f1']
            logger.info(f'AvP:      {ap:0.4f}')
            logger.info(f'F1:       {f1:0.4f}')


        test_performance_tracker.append(performance_result)
        logger.info('-------------------')

    data_container={}
    
    for iou in iou_thresholds:
        data_container[f'{iou}_ap'] = [tracker[str(iou)]['ap'] 
                                       for tracker in test_performance_tracker]
        data_container[f'{iou}_f1'] = [tracker[str(iou)]['f1'] 
                                       for tracker in test_performance_tracker]
    
    performance_df = pd.DataFrame(data=data_container)
    performance_df.to_csv(
        f'{directory}/results.csv'
    )

    # Precision-recall curves
    iou_array = []
    run_array = []
    precision_array = None
    recall_array = None

    for iou in iou_thresholds:
        for i_run, tracker in enumerate(test_performance_tracker):
            precision = tracker[str(iou)]['precision']
            recall = tracker[str(iou)]['recall']
            n_points = len(precision)

            if precision_array is None or recall_array is None:
                precision_array = precision
                recall_array = recall
            else:
                precision_array = np.concatenate((precision_array, precision))
                recall_array = np.concatenate((recall_array, recall))
            iou_array += n_points * [iou,]
            run_array += n_points * [i_run + 1,]

    precrec_data = {
        'iou': iou_array,
        'run_id': run_array,
        'precision': precision_array,
        'recall': recall_array
    }
    precrec_df = pd.DataFrame(data=precrec_data)
    precrec_df.to_csv(
        f'{directory}/precision_recall.csv'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_dir', type=str
    )
    parser.add_argument(
        '--event_proportion', type=float,
        default=0.2
    )
    parser.add_argument(
        '--n_runs', type=int,
        default=5
    )
    parser.add_argument(
        '--training_data', type=str
    )
    parser.add_argument(
        '--test_data', type=str
    )
    parser.add_argument(
        '--noise_data', type=str
    )

    args = parser.parse_args()
    main(args)