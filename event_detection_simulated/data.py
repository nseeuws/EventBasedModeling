from typing import Tuple, Optional

import numpy as np
import scipy.signal
import scipy.io
import einops
import wfdb
import h5py

import torch
from torch import Tensor
from torch.utils.data import Dataset

import encoding


SIGNAL_FREQUENCY = 200


def add_axis(array: np.ndarray) -> np.ndarray:
    return einops.rearrange(array, 't -> 1 t')


class FiniteDataset(Dataset):
    def __init__(
        self, signals: np.ndarray,
        locations: np.ndarray, durations: np.ndarray,
        labels: np.ndarray
    ) -> None:
        super().__init__()
        self.signals = torch.from_numpy(signals)
        self.locations = torch.from_numpy(locations)
        self.durations = torch.from_numpy(durations)
        self.labels = torch.from_numpy(labels) 
    
    def __len__(self):
        return len(self.signals)

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if torch.is_tensor(item):
            item = item.tolist()

        signal = self.signals[item]
        location = self.locations[item]
        duration = self.durations[item]
        label = self.labels[item]
        
        return signal, location, duration, label


class DataGenerator(Dataset):
    def __init__(
        self,
        ecg_examples: np.ndarray, noise_examples: np.ndarray,
        batch_stride: int, window_size: int, network_stride: int,
        max_n_events: int = 4, event_proportion: float = 0.8,
        max_duration: float = 0.3, min_duration: float = 0.05,
        min_snr: float = -6, max_snr: float = 6
    ):
        super().__init__()
        assert 0. <= event_proportion <= 1.0
        assert max_n_events > 0
        assert 0. < max_duration < 1.0
        assert 0. < min_duration < 1.0
        assert min_duration < max_duration
        assert min_snr < max_snr

        self.signals = ecg_examples
        self.noise_examples = noise_examples
        self.stride = batch_stride
        self.window_size = window_size
        self.network_stride = network_stride

        self.max_events = max_n_events
        self.proportion = event_proportion
        self.max_dur = max_duration
        self.min_dur = min_duration
        self.min_snr = min_snr
        self.max_snr = max_snr

        key_array = []
        for i, array in enumerate(self.signals):
            n = (len(array) - self.window_size) // self.stride
            for j in range(n):
                key_array.append([i, self.stride * j])
        self.key_array = np.asarray(key_array, dtype=np.uint32)

    def __len__(self):
        return self.key_array.shape[0]

    def __getitem__(self, item) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        recording, start_point = self.key_array[item]
        window = self.window_size

        signal = np.copy(self.signals[recording][
            start_point:start_point + window
        ])
        label = np.zeros(shape=(self.window_size,), dtype=np.uint8)

        # Event generation
        if torch.rand(size=(1,), device='cpu') < self.proportion:
            n_events = int(torch.randint(
                low=1, high=self.max_events + 1, size=(1,), device='cpu'
            ).numpy())

            edge = window // 10
            centers = torch.randint(
                low=edge, high=window - edge, size=(n_events,), device='cpu'
            ).numpy()
            minimum_duration = int(self.min_dur * window)
            maximum_duration = int(self.max_dur * window)
            durations = torch.randint(
                low=minimum_duration, high=maximum_duration, size=(n_events,), device='cpu'
            ).numpy()
            noise_recordings = torch.randint(
                low=0, high=len(self.noise_examples), size=(n_events,), device='cpu'
            ).numpy()
            snr_scalings = torch.rand(
                size=(n_events,), device='cpu'
            ).numpy()

            for center, duration, recording, snr_scaling in zip(
                centers, durations, noise_recordings, snr_scalings
            ):
                start = max(0, center - duration // 2)
                stop = min(window, center + duration // 2)
                actual_duration = stop - start

                noise_duration = len(self.noise_examples[recording])
                noise_start = int(
                    torch.randint(
                        low=0, high=noise_duration - actual_duration, size=(1,),
                        device='cpu'
                    ).numpy()
                )
                target = np.copy(self.noise_examples[recording][
                    noise_start:noise_start + actual_duration
                ]) # Have to make sure dimensions of `target` match up

                # Vary SNR
                snr_target = self.min_snr + \
                     snr_scaling * (self.max_snr - self.min_snr)

                signal_std = np.std(signal[start:stop])
                target_std = np.std(target)
                target_mean = np.mean(target)
                scaling = np.power(10., (
                    20 * np.log10(signal_std) - snr_target
                ) / 20.)

                target = (target - target_mean) * scaling / target_std

                # Add a smoothing window
                smoothing_window = scipy.signal.windows.tukey(M=stop - start)
                target *= smoothing_window

                signal[start:stop] = signal[start:stop] + target

                label[start:stop] = 1

        # Noise injection
        n_events = int(torch.randint(
            low=1, high=10, size=(1,), device='cpu'
        ))
        centers = torch.randint(
            low=0, high=window, size=(n_events,), device='cpu'
        )
        minimum_duration = 100
        maximum_duration = int(self.min_dur * window)
        durations = torch.randint(
            low=minimum_duration, high=maximum_duration, size=(n_events,), device='cpu'
        ).numpy()
        noise_recordings = torch.randint(
            low=0, high=len(self.noise_examples), size=(n_events,), device='cpu'
        ).numpy()
        snr_scalings = torch.rand(
            size=(n_events,), device='cpu'
        ).numpy()
        
        for center, duration, recording, snr_scaling in zip(
            centers, durations, noise_recordings, snr_scalings
        ):
            start = max(0, center - duration // 2)
            stop = min(window, center + duration // 2)
            actual_duration = stop - start

            noise_duration = len(self.noise_examples[recording])
            noise_start = int(
                torch.randint(
                low=0, high=noise_duration - actual_duration, size=(1,), device='cpu'
                ).numpy()
            )
            target = np.copy(self.noise_examples[recording][
                noise_start:noise_start + actual_duration
            ])
            snr_target = self.min_snr + snr_scaling * (self.max_snr - self.min_snr)
            signal_std = np.std(signal[start:stop])
            target_std = np.std(target)
            target_mean = np.mean(target)
            scaling = np.power(10., (
                20 * np.log10(signal_std) - snr_target
            ) / 20.)

            target = (target - target_mean) * scaling / target_std

            smoothing_window = scipy.signal.windows.tukey(M=stop - start)
            target *= smoothing_window

            signal[start:stop] = signal[start:stop] + target

        

        location, duration = encoding.get_target_maps(
            labels=[label,], stride=self.network_stride,
            duration_cutoff=int(self.max_dur * window)
        )

        location = add_axis(location[0])
        duration = add_axis(duration[0])
        signal = add_axis(signal)
        label = label[::self.network_stride]

        signal = torch.from_numpy(signal)
        location = torch.from_numpy(location)
        duration = torch.from_numpy(duration)
        label = torch.from_numpy(label)

        return signal, location, duration, label
    

def load_ecg_signals(
    training_path: str = '/volume1/scratch/nseeuws/ECG/cinc2017/train_val.hdf5',
    test_path: str = '/volume1/scratch/nseeuws/ECG/cinc2017/validation/' 
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Load training and validation data
    with h5py.File(training_path, 'r') as f:
        x_train_ls = []
        y_train_ls = []
        x_val_ls = []
        y_val_ls = []

        x_train_ds = f['x_train']
        y_train_ds = f['y_train']
        x_val_ds = f['x_val']
        y_val_ds = f['y_val']

        for i in range(len(x_train_ds)): # type: ignore
            x_train_ls.append(x_train_ds[i]) # type: ignore
            y_train_ls.append(y_train_ds[i]) # type: ignore

        for i in range(len(x_val_ds)): # type: ignore
            x_val_ls.append(x_val_ds[i]) # type: ignore
            y_val_ls.append(y_val_ds[i]) # type: ignore

    x_train = np.asarray(x_train_ls, dtype=object)
    y_train = np.asarray(y_train_ls, dtype=np.uint8)
    x_val = np.asarray(x_val_ls, dtype=object)
    y_val = np.asarray(y_val_ls, dtype=np.uint8)

    x_train = x_train[np.logical_or(y_train == 0, y_train == 1)]
    x_val = x_val[np.logical_or(y_val == 0, y_val == 1)]

    # Normalize train
    flat = np.hstack(x_train.flatten()) # type: ignore
    mean = np.mean(flat)
    std = np.std(flat)
    for i, signal in enumerate(x_train):
        x_train[i] = (signal - mean) / std
    
    # Normalize val
    flat = np.hstack(x_val.flatten()) # type: ignore
    mean = np.mean(flat)
    std = np.std(flat)
    for i, signal in enumerate(x_val):
        x_val[i] = (signal - mean) / std

    # Load test signals
    records_normal = 'RECORDS-normal'
    records_af = 'RECORDS-af'
    data = []

    with open(test_path + records_normal, 'r') as f:
        file_names = f.read().splitlines()
    for name in file_names:
        signal = scipy.io.loadmat(test_path+name+'.mat')
        data.append(np.asarray(signal['val'][0, :], dtype=np.float32))

    with open(test_path + records_af, 'r') as f:
        file_names = f.read().splitlines()
    for name in file_names:
        signal = scipy.io.loadmat(test_path+name+'.mat')
        data.append(np.asarray(signal['val'][0, :], dtype=np.float32))

    # Filtering and normalizing test signals
    SIGNAL_FREQUENCY = 200
    sos = scipy.signal.butter(N=5, Wn=[1, 50], btype='bandpass', output='sos', fs=SIGNAL_FREQUENCY)

    x_test_ls = []
    for signal in data:
        x_test_ls.append(scipy.signal.sosfiltfilt(sos=sos, x=signal))

    flat = np.concatenate(x_test_ls).ravel()
    mean = np.mean(flat)
    std = np.std(flat)
    del flat

    for i, signal in enumerate(x_test_ls):
        x_test_ls[i] = (signal - mean) / std

    # Setting up output
    x_test = np.asarray(x_test_ls, dtype=object)

    return x_train, x_val, x_test


def load_ecg_noise(noise_type: str, path: Optional[str] = None) -> np.ndarray:
    """Load ECG noise samples

    Args:
        noise_type (str): String describing the type of noise.
        One of "em", "ma", or "bw" for electrode motion, movement artefacts,
        or baseline wander respectively.

    Returns:
        List[np.ndarray]: List of example signals
    """
    assert noise_type == 'em' or noise_type == 'ma' or noise_type == 'bw'

    if path is not None:
        em_path = f'{path}/em'
        ma_path = f'{path}/ma'
        bw_path = f'{path}/bw'
    else:
        em_path = '/volume1/scratch/nseeuws/ECG/NoiseStressTest/em'
        ma_path = '/volume1/scratch/nseeuws/ECG/NoiseStressTest/ma'
        bw_path = '/volume1/scratch/nseeuws/ECG/NoiseStressTest/bw'

    if noise_type == 'em':
        path = em_path
    elif noise_type == 'ma':
        path = ma_path
    elif noise_type == 'bw':
        path = bw_path

    sos = scipy.signal.butter(N=5, Wn=[1, 50], btype='bandpass', output='sos', fs=SIGNAL_FREQUENCY)

    base, fields = wfdb.rdsamp(path)
    base_frequency = fields['fs']
    output = []
    for signal in base.T:
        filtered_signal = scipy.signal.sosfiltfilt(
            sos=sos, x=resample(
                signal=signal, base_freq=base_frequency,
                resample_freq=SIGNAL_FREQUENCY
            )
        )
        mean = filtered_signal.mean()
        std = filtered_signal.std()
        filtered_signal = (filtered_signal - mean) / std
        output.append(filtered_signal)

    return np.asarray(output)


def resample(signal: np.ndarray, base_freq: float, resample_freq: float) -> np.ndarray:
    """Resample 1D signal array

    Args:
        signal (np.ndarray): 1D array
        base_freq (float): Original frequency
        resample_freq (float): New frequency

    Returns:
        np.ndarray: Signal resampled at `resample_freq` frequency
    """
    time_steps = signal.shape[0]
    new_time_indices = np.arange(
        start=0, stop=time_steps / base_freq,
        step=1 / resample_freq
    )
    original_time_indices = np.arange(
        start=0, stop=time_steps / base_freq,
        step=1 / base_freq
    )

    interpolated_signal = np.interp(
        x=new_time_indices, xp=original_time_indices,
        fp=signal
    )

    return interpolated_signal


def build_finite_datasets(
        train_ds: DataGenerator,
        val_ds: DataGenerator, test_ds: DataGenerator,
        input_duration: int, network_stride: int,
) -> Tuple[
    FiniteDataset, FiniteDataset, FiniteDataset
]:
    # Test set
    n_samples = len(test_ds)
    signals = np.zeros(
        shape=(n_samples, 1, input_duration), dtype=np.float32
    )
    locations = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    durations = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    labels = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.int8
    )
    for i in range(n_samples):
        signal, location, duration, label = test_ds[i]
        signals[i, :, :] = signal.numpy()
        locations[i, :, :] = location.numpy()
        durations[i, :, :] = duration.numpy()
        labels[i, :] = label.numpy()
    finite_test_ds = FiniteDataset(
        signals=signals, locations=locations,
        durations=durations, labels=labels
    )

    # Validation
    n_samples = len(val_ds)
    signals = np.zeros(
        shape=(n_samples, 1, input_duration), dtype=np.float32
    )
    locations = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    durations = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    labels = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.int8
    )
    for i in range(n_samples):
        signal, location, duration, label = val_ds[i]
        signals[i, :, :] = signal.numpy()
        locations[i, :, :] = location.numpy()
        durations[i, :, :] = duration.numpy()
        labels[i, :] = label.numpy()
    finite_val_ds = FiniteDataset(
        signals=signals, locations=locations,
        durations=durations, labels=labels
    )

    # Training
    n_samples = len(train_ds)
    train_signal = np.zeros(
        shape=(n_samples, 1, input_duration), dtype=np.float32
    )
    train_location = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    train_duration = np.zeros(
        shape=(n_samples, 1, input_duration // network_stride),
        dtype=np.float32
    )
    train_label = np.zeros(
        shape=(n_samples, input_duration // network_stride),
        dtype=np.int8
    )
    train_change = np.zeros(
        shape=(n_samples, input_duration // network_stride),
        dtype=np.uint8
    )
 
    for i in range(n_samples):
        signal, location, duration, label = train_ds[i]
        train_signal[i, :, :] = signal.numpy()
        train_location[i, :, :] = location.numpy()
        train_duration[i, :, :] = duration.numpy()
        train_label[i, :] = label.numpy()

    
    # New dataset
    finite_train_ds = FiniteDataset(
        signals=train_signal, locations=train_location, durations=train_duration,
        labels=train_change
    )
    return finite_train_ds, finite_val_ds, finite_test_ds