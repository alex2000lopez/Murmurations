import torch
from torch.utils.data import Dataset
import numpy as np
import json
from scipy import signal
import os
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from collections import defaultdict

torch.backends.cudnn.benchmark = True

#from torchprofile import profile_macs

class MemmapDataset(Dataset):
    def __init__(self, descriptor_path, padding_handling="remove", interp_length=None, transform=None, filter=False):
        """
        Args:
            descriptor_path (str): Path to the descriptor JSON file (e.g., 'descriptor.json').
            padding_handling (str or float): How to handle np.inf padding values.
                - "remove" (default): Remove the padded np.inf values and return variable-length arrays.
                - A float: Replace any np.inf values with the given float.
            interp_length (int, optional): If provided, the ADC data (adc1 and adc2) will be
                first stripped of np.inf padding and then interpolated to this fixed length.
            transform (callable, optional): Optional transform to be applied on a sample.
            filter (bool, optional): Whether to apply a bandpass filter to the audio data.
        """
        # Load descriptor from JSON file.
        with open(descriptor_path, 'r') as f:
            self.descriptor = json.load(f)
        
        # Extract required parameters from the descriptor.
        self.audio_sampling_rate = self.descriptor['audio_sampling_rate']
        self.adc_sampling_rate = self.descriptor['adc_sampling_rate']
        self.audio_lowcut       = self.descriptor['audio_lowcut']
        self.audio_highcut      = self.descriptor['audio_highcut']
        self.adc_lowcut         = self.descriptor['adc_lowcut']
        self.adc_highcut        = self.descriptor['adc_highcut']
        self.max_audio_len      = self.descriptor['max_audio_len']
        self.max_adc_len        = self.descriptor['max_adc_len']
        self.n_segments         = self.descriptor['n_segments']
        self.raw_memmap_name    = self.descriptor['memmap_filename']
        self.dataset_mapping    = self.descriptor['dataset_mapping']

        # Make memmap path relative to the descriptor file location
        descriptor_dir = os.path.dirname(os.path.abspath(descriptor_path))
        self.memmap_filename = os.path.join(descriptor_dir, self.raw_memmap_name)

        # Rebuild the dtype from the descriptor.
        self.dtype = np.dtype([tuple(item) for item in self.descriptor['dtype']])

        # Open the memmap file in read-only mode using the number of segments from the descriptor.
        self.memmap = np.memmap(self.memmap_filename, dtype=self.dtype, mode='r', shape=(self.n_segments,))
        
        self.transform = transform
        self.padding_handling = padding_handling
        self.interp_length = interp_length
        self.filter = filter

    def __len__(self):
        return self.n_segments

    def __getitem__(self, index):
        # Retrieve the record from the memmap.
        row = self.memmap[index]
        
        # Convert fixed-size arrays to numpy arrays.
        audio_arr = np.array(row['audio'])
        adc1_arr = np.array(row['adc1'])
        adc2_arr = np.array(row['adc2'])
        
        # Process audio channel using the padding handling method.
        audio_arr = self._handle_padding(audio_arr, self.padding_handling)
        adc1_arr = self._handle_padding(adc1_arr, self.padding_handling)
        adc2_arr = self._handle_padding(adc2_arr, self.padding_handling)
        
        if self.filter:
            audio_arr = self.BPfilter(audio_arr, self.audio_sampling_rate, self.audio_lowcut, self.audio_highcut)
            adc1_arr = self.BPfilter(adc1_arr, self.adc_sampling_rate, self.adc_lowcut, self.adc_highcut)
            adc2_arr = self.BPfilter(adc2_arr, self.adc_sampling_rate, self.adc_lowcut, self.adc_highcut)

        # Process ADC channels.
        if self.interp_length is not None:
            audio_arr = self._interpolate_channel(audio_arr, self.interp_length)
            adc1_arr = self._interpolate_channel(adc1_arr, self.interp_length)
            adc2_arr = self._interpolate_channel(adc2_arr, self.interp_length)
        
        # Create a sample tuple.
        # Use .copy() to ensure the arrays have positive strides.
        sample = ( 
            int(row['id']), 
            torch.from_numpy(audio_arr.copy()).float(),  
            torch.from_numpy(adc1_arr.copy()).float(), 
            torch.from_numpy(adc2_arr.copy()).float(),  
        )
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _handle_padding(self, arr, mode):
        """
        Handle the np.inf padded values in the array.
        If mode is "remove", return the array with inf values removed.
        If mode is a float, replace inf values with that float.
        """
        if mode == "remove":
            return arr[~np.isinf(arr)]
        elif isinstance(mode, (int, float)):
            return np.where(np.isinf(arr), mode, arr)
        else:
            raise ValueError("Invalid padding_handling value. Use 'remove' or a float value.")

    def _interpolate_channel(self, arr, target_length):
        """
        Remove np.inf values from the array and linearly interpolate
        to the target_length.
        """
        # Remove padded inf values.
        valid = arr[~np.isinf(arr)]
        if len(valid) == 0:
            # If there is no valid data, return an array of zeros.
            return np.zeros(target_length, dtype=arr.dtype)
        # Generate new indices for interpolation.
        old_indices = np.arange(len(valid))
        new_indices = np.linspace(0, len(valid) - 1, target_length)
        return np.interp(new_indices, old_indices, valid)[:target_length]

    def get(self, field):
        """
        Return the value of the given descriptor field.
        For example, dataset.get("audio_sampling_rate") returns the audio sampling rate.
        """
        return self.descriptor.get(field, None)

    def id_to_dataset(self, id):
        """
        Return the dataset string for the given ID.
        """
        return self.dataset_mapping.get(str(id), "Unknown")

    def get_Nclasses(self):
        """
        Return the number of unique datasets in the dataset_mapping.
        """
        return len(set(self.dataset_mapping.values()))
    
    def BPfilter(self, data, fs, lowcut_hz=None, highcut_hz=None):
        """
        Apply a bandpass Butterworth filter to the input data.
        
        Parameters:
        data : array-like
            The input signal to filter
        fs : float
            Sampling frequency in Hz
        lowcut_hz : float, optional
            Lower cutoff frequency in Hz. If None, defaults to 20 Hz
        highcut_hz : float, optional
            Upper cutoff frequency in Hz. If None, defaults to fs/4 Hz
            
        Returns:
        array-like
            The filtered signal
        """
        # Default cutoff frequencies if not provided.
        if lowcut_hz is None:
            lowcut_hz = 20  # Default lower cutoff of 20 Hz
        if highcut_hz is None:
            highcut_hz = fs/4  # Default upper cutoff at quarter of sampling rate
        
        # Convert cutoff frequencies to normalized units (0 to 1).
        nyquist = fs / 2
        low = lowcut_hz / nyquist
        high = highcut_hz / nyquist
        
        # Create a 4th-order bandpass Butterworth filter.
        b, a = signal.butter(2, [low, high], btype='band')
        
        # Apply zero-phase filtering using filtfilt.
        filtered_data = signal.filtfilt(b, a, data)
        return filtered_data

    # def BPfilter(self, data, fs, lowcut_hz=None, highcut_hz=None):
    #     """
    #     Apply a bandpass Butterworth filter to the input data.

    #     For very short signals (len(data) <= padlen for filtfilt),
    #     fall back to lfilter to avoid ValueError.
    #     """
    #     # Default cutoff frequencies if not provided.
    #     if lowcut_hz is None:
    #         lowcut_hz = 20  # Default lower cutoff of 20 Hz
    #     if highcut_hz is None:
    #         highcut_hz = fs / 4  # Default upper cutoff at quarter of sampling rate

    #     # Convert cutoff frequencies to normalized units (0 to 1).
    #     nyquist = fs / 2.0
    #     low = lowcut_hz / nyquist
    #     high = highcut_hz / nyquist

    #     # 2nd-order bandpass Butterworth filter.
    #     b, a = signal.butter(2, [low, high], btype="band")

    #     # filtfilt needs len(data) > padlen, where padlen = 3 * (max(len(a), len(b)) - 1)
    #     padlen = 3 * (max(len(a), len(b)) - 1)

    #     if len(data) <= padlen:
    #         # Too short for filtfilt: use lfilter as a safe fallback
    #         # (or simply return data if you prefer no filtering here).
    #         return signal.lfilter(b, a, data)

    #     # Normal case: zero-phase filtering
    #     return signal.filtfilt(b, a, data)


class normalizer():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        id, audio, adc1, adc2 = sample
        audio = (audio - self.mean[0]) / self.std[0]
        adc1 = (adc1 - self.mean[1]) / self.std[1]
        adc2 = (adc2 - self.mean[1]) / self.std[1]
        return id, audio, adc1, adc2
    


def plot_random_example(dataset):

    rng_index = np.random.randint(0, len(dataset))
    sample = dataset[rng_index]

    sample_id, audio, adc1, adc2 = sample  # unpack the tuple

    # Convert tensors to numpy arrays for plotting
    audio_np = audio.numpy()
    adc1_np = adc1.numpy()
    adc2_np = adc2.numpy()

    # Create a figure with three subplots for audio, ADC1, and ADC2.
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    axs[0].plot(audio_np)
    axs[0].set_title(f"Audio Sample (ID: {sample_id}=>{dataset.id_to_dataset(sample_id)})")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Amplitude")

    axs[1].plot(adc1_np)
    axs[1].set_title(f"ADC1 Sample (n={len(adc1_np)} samples)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Amplitude")

    axs[2].plot(adc2_np)
    axs[2].set_title(f"ADC2 Sample (n={len(adc2_np)} samples)")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


# Pytorch models
    
def computeModelSize(model):
    params = sum(p.numel() for p in model.parameters())
    model_size = params * 4 / (1024 ** 2)  # Convert to MB assuming 32-bit (4 bytes) precision
    return model_size

# def computeModelStats(model, input_shape):
#     dummy_input = torch.randn(input_shape)
#     macs = profile_macs(model, dummy_input)
#     params = sum(p.numel() for p in model.parameters())
#     model_size = params * 4 / (1024 ** 2)  # Convert to MB assuming 32-bit (4 bytes) precision
#     print(f"MACs: {macs}")
#     print(f"Parameters: {params}")
#     print(f"Model Size: {model_size:.2f} MB")

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AudioResNet(nn.Module):
    def __init__(self, input_length, input_dim,output_length):
        super(AudioResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 128, 2)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 512, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_length)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ResNet(nn.Module):
    def __init__(self, input_length, input_dim,output_length):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer( 64, 128, 4)
        self.layer2 = self._make_layer(128, 256, 2, stride=2)
        self.layer3 = self._make_layer(256, 512, 2, stride=2)
        self.layer4 = self._make_layer(512, 512, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_length)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SmallResNet(nn.Module):
    def __init__(self, input_length, input_dim, output_length):
        super(SmallResNet, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ELU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(32, 64, 4)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 4, stride=2)
        self.layer4 = self._make_layer(256, 512, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_length)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class InceptionBlock(nn.Module):
    """
    An Inception-style block for 1D signals.
    Each block includes:
      1) Optional bottleneck conv (1x1) to reduce input channels if needed
      2) Three parallel conv branches with different kernel sizes
      3) A parallel max-pool branch
      4) Concatenation of all branches
      5) A 1x1 'linear' conv to reduce channels back to 'out_channels'
      6) Optional skip connection if use_residual=True
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list = [9, 19, 39],
                 bottleneck_channels: int = 32,
                 use_residual: bool = True):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (post concatenation reduction)
            kernel_sizes: Sizes for the parallel convolutions
            bottleneck_channels: Bottleneck channels for the optional 1x1 conv
            use_residual: Whether to add a residual (skip) connection
        """
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual

        # 1) Bottleneck if input channels are larger than the bottleneck size
        if in_channels > bottleneck_channels:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
            self.bottleneck_bn = nn.BatchNorm1d(bottleneck_channels)
            conv_in = bottleneck_channels
        else:
            self.bottleneck = None
            conv_in = in_channels

        # 2) Three parallel conv branches with different kernel sizes
        self.branch1 = nn.Sequential(
            nn.Conv1d(conv_in, out_channels, kernel_size=kernel_sizes[0],
                      padding=kernel_sizes[0] // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(conv_in, out_channels, kernel_size=kernel_sizes[1],
                      padding=kernel_sizes[1] // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(conv_in, out_channels, kernel_size=kernel_sizes[2],
                      padding=kernel_sizes[2] // 2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True)
        )

        # 3) Pooling branch
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True)
        )

        # 4) 1x1 'linear' conv to reduce from 4*out_channels -> out_channels
        self.conv_linear = nn.Conv1d(4 * out_channels, out_channels, kernel_size=1, bias=False)
        self.conv_linear_bn = nn.BatchNorm1d(out_channels)

        # If using a residual connection but in_channels != out_channels, we align dims
        self.skip_conv = None
        if use_residual and in_channels != out_channels:
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.final_activation = nn.ELU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bottleneck if configured
        if self.bottleneck is not None:
            x_bottleneck = self.bottleneck_bn(self.bottleneck(x))
        else:
            x_bottleneck = x

        # Parallel branches
        out1 = self.branch1(x_bottleneck)
        out2 = self.branch2(x_bottleneck)
        out3 = self.branch3(x_bottleneck)
        out4 = self.branch4(x)

        # Concatenate
        merged = torch.cat([out1, out2, out3, out4], dim=1)   # (B, 4*out_channels, L)

        # Linear conv to get back to out_channels
        merged = self.conv_linear_bn(self.conv_linear(merged))  # (B, out_channels, L)

        # Residual/skip connection
        if self.use_residual:
            skip = x
            if self.skip_conv is not None:
                skip = self.skip_conv(x)
            merged = self.final_activation(merged + skip)
        else:
            merged = self.final_activation(merged)

        return merged

class InceptionTime(nn.Module):
    """
    An InceptionTime model that stacks several InceptionBlocks in sequence
    and then applies global average pooling + a final linear classifier.
    """
    def __init__(self,
                 input_length: int,
                 input_dim: int,
                 output_length: int,
                 num_filters: int = 32,
                 bottleneck_channels: int = 32,
                 num_inception_blocks: int = 3):
        """
        Args:
            input_length: Unused directly here, but kept for parity with other models.
            input_dim: Number of input channels (e.g., 2 if you have 2 ADC channels).
            output_length: Number of output classes.
            num_filters: Base number of filters for each inception branch
            bottleneck_channels: Bottleneck size for optional 1x1 conv
            num_inception_blocks: How many Inception blocks to stack
        """
        super(InceptionTime, self).__init__()

        blocks = []
        current_in_channels = input_dim
        for _ in range(num_inception_blocks):
            block = InceptionBlock(
                in_channels=current_in_channels,
                out_channels=num_filters,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=bottleneck_channels,
                use_residual=True
            )
            blocks.append(block)
            current_in_channels = num_filters

        self.inception_blocks = nn.Sequential(*blocks)

        # Adaptive average pool to (B, C, 1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully-connected classification layer
        self.fc = nn.Linear(num_filters, output_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through stacked Inception blocks
        x = self.inception_blocks(x)  # (B, num_filters, L)

        # Global average pool
        x = self.global_pool(x)       # (B, num_filters, 1)
        x = x.squeeze(-1)            # (B, num_filters)

        # Fully-connected classification
        x = self.fc(x)               # (B, output_length)
        return x
    

# Training functions
    
def train_epoch(
    model,
    dataloader,
    epochs,                 # we will always pass epochs=1
    learning_rate,
    optimizer=None,
    loss_function=None,
    scheduler=None,
    device="cpu",
    verbose=0,
    log_interval=100,
    logger=None,
    amodel=None,
    use_kd=True,
    temperature=2.0,
    alpha_kd=0.9,
    use_amp=True,
):
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    
    if optimizer is None:
        if learning_rate is None:
            raise ValueError("learning_rate must be provided when optimizer is None")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device(device)
    model.train()

    use_kd = use_kd and (amodel is not None)
    if use_kd:
        amodel.eval()

    kd_loss_fn = nn.KLDivLoss(reduction="batchmean") if use_kd else None

    use_amp = use_amp and (device.type == "cuda")
    scaler = GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    for batch_idx, (ids, audio, adc1, adc2) in enumerate(dataloader):
        audio = audio.to(device, non_blocking=True)
        adc1 = adc1.to(device, non_blocking=True)
        adc2 = adc2.to(device, non_blocking=True)
        ids = ids.to(device, non_blocking=True)

        adc = torch.stack((adc1, adc2), dim=1)
        adc = adc + torch.randn_like(adc) * 0.04

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            outputs = model(adc)
            loss1 = loss_function(outputs, ids)
            loss = loss1

            if use_kd:
                with torch.no_grad():
                    teacher_outputs = amodel(audio.unsqueeze(1))

                T = temperature
                log_p = F.log_softmax(outputs / T, dim=1)
                q = F.softmax(teacher_outputs / T, dim=1)
                loss2 = kd_loss_fn(log_p, q) * (T * T)
                loss = loss1 + alpha_kd * loss2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == ids).sum().item()
        total += ids.size(0)

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)

    if verbose >= 2:
        print(f"[TRAIN] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    if logger is not None and verbose != 0:
        logger.log({"split": "train", "avg_loss": avg_loss, "accuracy": accuracy})

    # keep same API: list of one (loss, acc) tuple
    return [(avg_loss, accuracy)]

def test_epoch(
    model,
    dataloader,
    epochs=1,          # always 1
    loss_function=None,
    device="cpu",
    verbose=0,
    log_interval=100,
    logger=None,
):
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    device = torch.device(device)
    model.eval()

    use_amp = (device.type == "cuda")

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (ids, audio, adc1, adc2) in enumerate(dataloader):
            ids = ids.to(device, non_blocking=True)
            adc1 = adc1.to(device, non_blocking=True)
            adc2 = adc2.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)

            adc = torch.stack((adc1, adc2), dim=1)

            with autocast("cuda", enabled=use_amp):
                outputs = model(adc)
                loss = loss_function(outputs, ids)

            total_loss += loss.item()
            num_batches += 1

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == ids).sum().item()
            total += ids.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)

    if verbose >= 2:
        print(f"[TEST]  Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    if logger is not None and verbose != 0:
        logger.log({"split": "test", "avg_loss": avg_loss, "accuracy": accuracy})

    return [(avg_loss, accuracy)]



# Data preparation

def stratified_split(dataset, train_to_val_ratio, label_index=0, seed=42):
    """
    Split dataset into train/val keeping (approximately) the same
    class distribution in both splits.
    Assumes dataset[i][label_index] is the class label (int or 0D tensor).
    """
    device_generator = torch.Generator().manual_seed(seed)

    # 1) Collect labels
    labels = []
    for i in range(len(dataset)):
        y = dataset[i][label_index]
        if isinstance(y, torch.Tensor):
            y = y.item()
        labels.append(int(y))

    # 2) Group indices by class
    indices_per_class = defaultdict(list)
    for idx, y in enumerate(labels):
        indices_per_class[y].append(idx)

    train_indices = []
    val_indices = []

    # 3) For each class, split its indices according to the ratio
    for cls, idxs in indices_per_class.items():
        idxs_tensor = torch.tensor(idxs)
        perm = idxs_tensor[torch.randperm(len(idxs), generator=device_generator)].tolist()

        n_total = len(perm)
        n_train = int(n_total * train_to_val_ratio)

        # Make sure we don't swallow the entire class into train
        # so that validation still sees at least 1 example if possible
        if n_train == n_total and n_total > 1:
            n_train = n_total - 1

        cls_train = perm[:n_train]
        cls_val = perm[n_train:]

        train_indices.extend(cls_train)
        val_indices.extend(cls_val)

    # 4) Shuffle final train/val index lists
    train_indices = torch.tensor(train_indices)
    val_indices = torch.tensor(val_indices)

    train_indices = train_indices[torch.randperm(len(train_indices), generator=device_generator)].tolist()
    val_indices = val_indices[torch.randperm(len(val_indices), generator=device_generator)].tolist()

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    return train_dataset, val_dataset


def prepare_datasets(dataset, train_to_val_ratio, batch_size, stratified=True, num_workers = 0, seed=42):

    if stratified:
        # Stratified by class id (assumed to be dataset[i][0])
        train_dataset, val_dataset = stratified_split(
            dataset,
            train_to_val_ratio=train_to_val_ratio,
            label_index=0,
            seed=seed,
        )
    else:
        # Old behavior: random, not stratified
        train_size = int(train_to_val_ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Use pinned memory when running on CUDA for faster host-to-device copies
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, train_loader, val_loader


from collections import Counter
import torch

def count_class_members(train_dataset, val_dataset, label_index=0):
    """
    Count how many samples of each class are in train and val datasets.

    Args:
        train_dataset: torch.utils.data.Dataset or Subset
        val_dataset:   torch.utils.data.Dataset or Subset
        label_index:   index in dataset[i] where the label is (default: 0)

    Returns:
        train_counts, val_counts: dict[class_id -> count]
    """
    def _count(ds):
        c = Counter()
        for i in range(len(ds)):
            y = ds[i][label_index]
            if isinstance(y, torch.Tensor):
                y = y.item()
            c[int(y)] += 1
        return c

    train_counts = _count(train_dataset)
    val_counts = _count(val_dataset)

    # Pretty print
    all_classes = sorted(set(train_counts.keys()) | set(val_counts.keys()))
    print("Class | train | val")
    print("--------------------")
    for cls in all_classes:
        t = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        print(f"{cls:5d} | {t:5d} | {v:3d}")

    return train_counts, val_counts


def train_model(
    descriptor_path,
    TOTAL_EPOCHS,
    start_epoch,
    verbosity,
    model,
    amodel,
    train_loader,
    val_loader,
    criterion,
    scheduler,
    optimizer,
    global_device,
    use_kd=True,
    use_amp=True,
    temperature=2.0,
    alpha_kd=0.9,
    save_model=True,
):

    device = torch.device(global_device)

    model.to(device)
    if amodel is not None and use_kd:
        amodel.to(device)

    train_stats = []
    test_stats = []

    pbar = tqdm(
        range(TOTAL_EPOCHS),
        desc="Epochs",
        disable=(verbosity == 0),
    )

    for epoch in pbar:
        # TRAIN
        res_train = train_epoch(
            model=model,
            dataloader=train_loader,
            epochs=1,
            learning_rate=None,
            optimizer=optimizer,
            loss_function=criterion,
            scheduler=scheduler,
            device=device,
            verbose=verbosity,  # 0: silent, 1: tqdm-only, 2+: also prints inside
            logger=None,
            amodel=amodel,
            use_kd=use_kd,
            temperature=temperature,
            alpha_kd=alpha_kd,
            use_amp=use_amp,
        )
        train_stats.extend(res_train)
        train_loss, train_acc = res_train[-1]

        # VAL
        res_val = test_epoch(
            model=model,
            dataloader=val_loader,
            epochs=1,
            loss_function=criterion,
            device=device,
            verbose=verbosity,
            logger=None,
        )
        test_stats.extend(res_val)
        val_loss, val_acc = res_val[-1]

        if verbosity >= 1:
            pbar.set_postfix(
                train_loss=f"{train_loss:.3f}",
                val_loss=f"{val_loss:.3f}",
                train_acc=f"{train_acc:.1f}",
                val_acc=f"{val_acc:.1f}",
            )

        if verbosity >= 2:
            print(
                f"[EPOCH {epoch+1}/{TOTAL_EPOCHS}] "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
            )

    if save_model:
        base_path, _ = os.path.splitext(descriptor_path)
        checkpoint_path = base_path + "_checkpoint_latest.pth"

        checkpoint = {
            "epoch": TOTAL_EPOCHS if start_epoch == 1 else start_epoch,           
            "model_state": model.state_dict(),
            "teacher_state": amodel.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_stats": train_stats,
            "test_stats": test_stats,
        }
        torch.save(checkpoint, checkpoint_path)

    return train_stats, test_stats, model, amodel

# Plotting functions

def plot_loss_accuracy(train_stats, test_stats, descriptor_path):

    # plot the training and testing loss and accuracy
    train_loss = [stat[0] for stat in train_stats]
    train_acc = [stat[1] for stat in train_stats]
    test_loss = [stat[0] for stat in test_stats]
    test_acc = [stat[1] for stat in test_stats]
    x_test = np.linspace(0, len(train_loss), len(test_loss))

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(x_test, test_loss, label='Test Loss')
    plt.axhline(np.min(test_loss), color='r', linestyle='--', label='Test Start')
    plt.text(0, np.min(test_loss), f"{np.min(test_loss):.4f}", va='center', ha='left')
    plt.title('Loss')
    plt.grid()
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(x_test, test_acc, label='Test Accuracy')
    plt.axhline(np.max(test_acc), color='r', linestyle='--', label='Test Start')
    plt.text(0, np.max(test_acc), f"{np.max(test_acc):.2f}%", va='center', ha='left')
    plt.title('Accuracy')
    plt.grid()
    plt.legend()

    base_path, _ = os.path.splitext(descriptor_path)
    loss_accuracy_save_path = base_path + "_loss_accuracy.png"

    if descriptor_path is not None:
        plt.savefig(loss_accuracy_save_path)
    else:
        plt.show()

def plot_confusion_matrix(model, dataset, dataloader, device, descriptor_path):

    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    output_length = dataset.get_Nclasses()
    
    with torch.no_grad():
        for ids, audio, adc1, adc2 in tqdm(dataloader):
            adc1 = adc1.to(device)
            adc2 = adc2.to(device)
            ids = ids.to(device)
            
            adc = torch.stack((adc1, adc2), dim=1)
            outputs = model(adc)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(ids.cpu().numpy())

    # Generate the confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Replace IDs with dataset names
    label_names = [dataset.id_to_dataset(label) for label in range(output_length)]
    pred_names = [dataset.id_to_dataset(pred) for pred in range(output_length)]

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=pred_names, yticklabels=label_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    base_path, _ = os.path.splitext(descriptor_path)
    confusion_matrix_path = base_path + "_confusion_matrix.svg"

    if descriptor_path is not None:
        plt.savefig(confusion_matrix_path)
    else:
        plt.show()


import numpy as np
from torch.utils.data import DataLoader

def evaluate_model_on_descriptor(
    model,
    descriptor_path,
    input_length,
    device,
    batch_size=32,
    num_workers=0,
    filter=True,
    loss_function=None,
    plot_cm=True,
):
    """
    Evaluate a trained model on an unseen dataset described by descriptor_path.

    - Builds a MemmapDataset from the descriptor.
    - Applies normalization using the stats stored in the descriptor.
    - Creates a DataLoader.
    - Runs the model and returns (avg_loss, accuracy).
    - Optionally plots the confusion matrix using plot_confusion_matrix
      so it looks identical to your existing plots.

    Args:
        model:          Trained PyTorch model (expects input (B, 2, input_length)).
        descriptor_path: Path to the *_descriptor.json for the unseen dataset.
        input_length:   Length to which ADC channels are interpolated.
        device:         Torch device ('cpu', 'cuda:0', torch.device(...), etc.).
        batch_size:     Batch size for evaluation DataLoader.
        num_workers:    DataLoader workers.
        filter:         Whether to apply BP filtering in MemmapDataset.
        loss_function:  Optional loss; defaults to CrossEntropyLoss.
        plot_cm:        If True, calls plot_confusion_matrix with the new dataset.

    Returns:
        avg_loss, accuracy (in %)
    """
    device = torch.device(device)

    # 1) Build dataset from descriptor
    eval_dataset = MemmapDataset(
        descriptor_path,
        padding_handling="remove",
        interp_length=input_length,
        transform=None,      # we set it after we build the normalizer
        filter=filter,
    )

    # 2) Normalizer using descriptor stats
    mean = [eval_dataset.get("audio_mean"), eval_dataset.get("adc_mean")]
    std  = [eval_dataset.get("audio_std"),  eval_dataset.get("adc_std")]
    eval_dataset.transform = normalizer(mean=mean, std=std)

    # 3) DataLoader
    pin_memory = (device.type == "cuda")
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 4) Evaluation loop
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()

    model.to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    with torch.no_grad():
        for ids, audio, adc1, adc2 in tqdm(eval_loader, desc="Evaluating"):
            ids = ids.to(device, non_blocking=True)
            adc1 = adc1.to(device, non_blocking=True)
            adc2 = adc2.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)

            adc = torch.stack((adc1, adc2), dim=1)

            outputs = model(adc)
            loss = loss_function(outputs, ids)

            total_loss += loss.item()
            num_batches += 1

            _, preds = torch.max(outputs, 1)
            correct += (preds == ids).sum().item()
            total += ids.size(0)

    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)

    print(f"[EVAL] Descriptor: {descriptor_path}")
    print(f"[EVAL] Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}% "
          f"({correct}/{total})")

    # 5) Confusion matrix plot, using your existing helper so style stays identical
    if plot_cm:
        # This assumes you already have plot_confusion_matrix(model, dataset, loader, device, descriptor_path)
        plot_confusion_matrix(model, eval_dataset, eval_loader, device, descriptor_path)

    return avg_loss, accuracy