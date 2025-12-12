import os
import sys
from DataLoader import H5DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import librosa
import librosa.display
import json
from scipy import signal
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from umap import UMAP
import math

# Function to load data from an H5 file
def get_file_path(identifier, person):

    data_dir = '../data/'
    # Available files
    files = os.listdir(data_dir)

    # Define the files to ignore
    files_to_ignore = ['Icon\r']

    filtered_files = [file for file in files if file not in files_to_ignore]
    print(filtered_files)
    
    # Make the search case-insensitive
    person = person.lower()
    identifier = identifier.lower()
    
    # Search for matching file
    for file in files:
        file_lower = file.lower()
        if identifier in file_lower and person in file_lower and file_lower.endswith(".h5"):
            print("found " + data_dir + file)
            return data_dir + file
        
    # If no match is found, return a descriptive message
    error_message = f"No recording found for identifier '{identifier}' and person '{person}'"
    print(error_message)
    return None


def inspect_h5_dataset(loader, AUDIO_SAMPLE_RATE=48000, ADC_SAMPLE_RATE=8000):
    """
    Inspect the first dataset in an H5DataLoader instance and print summary info.

    Args:
        loader (H5DataLoader): Instance of your H5DataLoader class.
    """
    datasets = loader.list_datasets()
    print("Available datasets:", datasets)

    if not datasets:
        print("No datasets found.")
        return None

    evaldata = loader.load_dataset(datasets[0])

    print("Eval Audio:", len(evaldata["audio_data"]))
    print("Eval ADC channels:", evaldata["adc_data"].keys())
    print("Eval ADC ch 1:", evaldata["adc_data"][1].shape)
    print("Eval ADC ch 3:", evaldata["adc_data"][3].shape)

    N_audio_samples = len(evaldata["audio_data"])
    N_adc_samples = len(evaldata["adc_data"][1]) + len(evaldata["adc_data"][3])

    print("Eval Audio samples:", N_audio_samples)
    print("Eval ADC samples:", N_adc_samples)

    ratio = N_audio_samples / N_adc_samples
    expected_ratio = AUDIO_SAMPLE_RATE / ADC_SAMPLE_RATE
    
    print("Audio to ADC ratio:", ratio)
    print("Expected ratio:", expected_ratio)

    # Return summary in case you need it programmatically
    return {
        "datasets": datasets,
        "N_audio_samples": N_audio_samples,
        "N_adc_samples": N_adc_samples,
        "adc_channels": list(evaldata["adc_data"].keys())
    }


def short_time_energy_segmentation(audio_samples, sr, 
                                   frame_duration=0.02,    # 20 ms 
                                   hop_duration=0.01,      # 10 ms
                                   smoothing_window=5,     # in frames
                                   energy_quantile=0.2,    # quantile for threshold
                                   min_silence_frames=3,   # minimum consecutive silent frames for a silence region
                                   min_voiced_frames=3     # minimum consecutive voiced frames for a speech region
                                  ):
    """
    Perform Short-Time Energy + Adaptive Thresholding segmentation.
    
    Parameters
    ----------
    audio_samples : 1D np.array
        Audio signal samples.
    sr : int
        Sample rate of the audio (samples/sec).
    frame_duration : float
        Duration of each frame in seconds (default 20 ms).
    hop_duration : float
        Hop (overlap) between frames in seconds (default 10 ms).
    smoothing_window : int
        Number of frames for smoothing the energy contour.
    energy_quantile : float
        Which quantile of the energy distribution to use for threshold. 
        For example, 0.2 = 20th percentile.
    min_silence_frames : int
        Minimum consecutive frames that must be below threshold to be considered silence.
    min_voiced_frames : int
        Minimum consecutive frames that must be above threshold to be considered speech.
    
    Returns
    -------
    segments : list of (seg_start, seg_end)
        Detected voiced (speech) segments in sample indices.
    ste : np.array
        Short-time energy array (one value per frame).
    frame_times : np.array
        Time (seconds) at the center of each frame.
    threshold : float
        Energy threshold used for classification.
    """

    # -------------------------
    # 1) Frame Setup
    # -------------------------
    frame_size = int(frame_duration * sr)
    hop_size   = int(hop_duration   * sr)

    # Make sure audio is 1D (mono)
    audio_samples = np.asarray(audio_samples).flatten()
    n_samples = len(audio_samples)

    # -------------------------
    # 2) Compute Short-Time Energy
    # -------------------------
    energies = []
    frame_times = []  # store center time of each frame
    idx = 0
    while idx + frame_size <= n_samples:
        frame = audio_samples[idx:idx + frame_size]
        ste_value = np.sum(frame**2) / frame_size
        energies.append(ste_value)

        # Time stamp for the center of the frame
        frame_center = (idx + frame_size/2.0) / sr
        frame_times.append(frame_center)

        idx += hop_size

    energies = np.array(energies)
    frame_times = np.array(frame_times)

    # -------------------------
    # 3) Smooth the Energy
    # -------------------------
    # Simple moving average over 'smoothing_window' frames
    kernel = np.ones(smoothing_window) / smoothing_window
    ste_smoothed = np.convolve(energies, kernel, mode='same')

    # -------------------------
    # 4) Adaptive Threshold
    # -------------------------
    # Example: pick the 'energy_quantile' percentile of the STE distribution
    # to serve as a baseline. Then add a factor if needed.
    threshold_value = np.quantile(ste_smoothed, energy_quantile)
    # Optionally scale that threshold:
    threshold_value *= 2.0  # e.g., multiply by 2

    # -------------------------
    # 5) Classify Frames as Voiced / Unvoiced
    # -------------------------
    voiced_frames = ste_smoothed >= threshold_value
    
    # We might want to "cleanup" tiny silences or tiny voiced pockets:
    # * Merge short voiced segments
    # * Merge short silence segments

    # We'll do a simple pass that merges short runs below/above threshold
    def merge_labels(labels, min_count, target_value):
        """
        Merge short segments of 'target_value' if they are below min_count.
        For example, merge short runs of 1's among 0's if min_count is 3.
        """
        # labels is boolean array. True=voiced, False=silence
        # We want to ensure runs of 'target_value' have at least min_count frames.
        # If not, convert them to the opposite label.
        labels_int = labels.astype(int)
        merged = labels_int.copy()
        
        i = 0
        while i < len(labels_int):
            val = labels_int[i]
            run_start = i
            while i < len(labels_int) and labels_int[i] == val:
                i += 1
            run_end = i  # one past the end
            
            run_length = run_end - run_start
            
            if val == target_value and run_length < min_count:
                # Flip them to the opposite value
                merged[run_start:run_end] = 1 - target_value
        
        return merged.astype(bool)

    # Merge short voiced frames
    voiced_frames = merge_labels(voiced_frames, min_voiced_frames, target_value=True)
    # Merge short silence frames
    voiced_frames = merge_labels(voiced_frames, min_silence_frames, target_value=False)

    # -------------------------
    # 6) Convert Frames to Time Segments
    # -------------------------
    segments = []
    in_segment = False
    seg_start = None

    for i, vf in enumerate(voiced_frames):
        if vf and not in_segment:
            # start of a segment
            in_segment = True
            seg_start = i
        elif not vf and in_segment:
            # end of a segment
            in_segment = False
            seg_end = i - 1
            # Convert frame indices to sample indices
            # We'll map to the center time of the first and last frames,
            # then expand to sample indices.
            seg_start_time = frame_times[seg_start] - (frame_duration/2)
            seg_end_time = frame_times[seg_end] + (frame_duration/2)
            sample_start = int(max(seg_start_time, 0) * sr)
            sample_end   = int(min(seg_end_time, n_samples / sr) * sr)
            segments.append((sample_start, sample_end))

    # If ended in a voiced segment
    if in_segment:
        seg_end = len(voiced_frames) - 1
        seg_start_time = frame_times[seg_start] - (frame_duration/2)
        seg_end_time   = frame_times[seg_end] + (frame_duration/2)
        sample_start   = int(max(seg_start_time, 0) * sr)
        sample_end     = int(min(seg_end_time, n_samples / sr) * sr)
        segments.append((sample_start, sample_end))

    return segments, ste_smoothed, frame_times, threshold_value


# -----------------------------------------------------
# 1) Run Short-Time Energy + Threshold Segmentation
# -----------------------------------------------------
# segments, ste, frame_times, threshold_value = short_time_energy_segmentation(
#     evaldata["audio_data"], AUDIO_SAMPLE_RATE,
#     frame_duration=0.04,
#     hop_duration=0.1,
#     smoothing_window=6,
#     energy_quantile=0.15,
#     min_silence_frames=1,
#     min_voiced_frames=2
# )

# # -----------------------------------------------------

# # 2) Visualization
# # -----------------------------------------------------

# # 2A) Plot Short-Time Energy & Threshold
# plt.figure(figsize=(12, 4))
# plt.plot(frame_times, ste, label="Short-Time Energy")
# plt.axhline(threshold_value, linestyle='--', label=f"Threshold={threshold_value:.4f}")
# plt.title("Short-Time Energy vs. Time")
# plt.xlabel("Time (sec)")
# plt.ylabel("Energy")
# plt.legend()
# plt.grid()
# plt.show()

# # 2B) Plot the Original Waveform & Highlight Detected Segments
# plt.figure(figsize=(12, 4))
# times = np.arange(len(evaldata["audio_data"])) / AUDIO_SAMPLE_RATE
# plt.plot(times, evaldata["audio_data"], label="Waveform")
# # Overlay detected segments as colored spans
# for (start_samp, end_samp) in segments:
#     plt.axvspan(start_samp/AUDIO_SAMPLE_RATE, end_samp/AUDIO_SAMPLE_RATE, color='green', alpha=0.3)
# plt.title("Waveform with Detected 'Speech' Segments")
# plt.xlabel("Time (sec)")
# plt.ylabel("Amplitude")
# plt.xlim([0, max(times)])
# plt.legend()
# plt.grid()
#plt.show()
#------------------------------------------------------

def compute_and_plot_ste(loader, audio_sample_rate, segmenter_func, **segmenter_kwargs):
    """
    Analyze all datasets in an H5DataLoader, segment audio using the short-time energy,
    and plot waveform + STE for each dataset.

    Args:
        loader (H5DataLoader): Your dataset loader instance.
        audio_sample_rate (float): Sampling rate of the audio.
        segmenter_func (callable): Segmentation function taking
            (audio_data, sample_rate, **kwargs) and returning
            (segments, ste, frame_times, threshold_value).
        **segmenter_kwargs: Optional keyword arguments passed directly to segmenter_func.

    Returns:
        tuple:
            - dataset_index (dict[str, list[tuple[int, int]]]):
                Mapping dataset name → list of (start, end) segments.
            - results (dict[str, dict]):
                Mapping dataset name → detailed results:
                {
                    "segments": list of (start, end),
                    "ste": np.array,
                    "frame_times": np.array,
                    "threshold_value": float
                }
    """
    results = {}
    dataset_index = {}
    datasets = loader.list_datasets()

    if not datasets:
        print("No datasets found.")
        return dataset_index, results

    # Default segmentation parameters (can be overridden)
    default_params = {
        "frame_duration": 0.3,
        "hop_duration": 0.1,
        "smoothing_window": 4,
        "energy_quantile": 0.35,
        "min_silence_frames": 1,
        "min_voiced_frames": 3,
    }
    params = {**default_params, **segmenter_kwargs}

    for dataset in datasets:
        print(f"Dataset: {dataset}")
        data = loader.load_dataset(dataset)
        audio_data = data["audio_data"]

        # --- Run segmentation ---
        segments, ste_smoothed, frame_times, threshold_value = segmenter_func(
            audio_data, audio_sample_rate, **params
        )

        # Store results
        dataset_index[dataset] = segments
        results[dataset] = {
            "segments": segments,
            "ste": ste_smoothed,
            "frame_times": frame_times,
            "threshold_value": threshold_value,
        }

        # --- Plot results ---
        plt.figure(figsize=(20, 10))
        plt.subplot(2, 1, 1)

        times = np.arange(len(audio_data)) / audio_sample_rate
        plt.plot(times, audio_data, label="Waveform")
        y_text = max(audio_data)

        for i, (start_samp, end_samp) in enumerate(segments):
            plt.axvspan(start_samp / audio_sample_rate, end_samp / audio_sample_rate,
                        color='green', alpha=0.3)
            plt.text((start_samp + end_samp) / (2 * audio_sample_rate), y_text,
                     f"{i+1}", fontsize=8, color='black', ha='center', va='center')

        plt.title(f"Waveform for '{dataset}'")
        plt.xlabel("Time (sec)")
        plt.ylabel("Amplitude")
        plt.xlim([0, max(times)])
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(frame_times, ste_smoothed, label="Short-Time Energy")
        plt.axhline(threshold_value, linestyle='--',
                    label=f"Threshold={threshold_value:.4f}")
        plt.title(f"Short-Time Energy for '{dataset}'")
        plt.xlabel("Time (sec)")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid()

    return dataset_index, results


# def get_exclude_config(person, identifier):
#     """
#     Returns the exclude_indexes and exclude_datasets based on person and identifier.
    
#     Args:
#         person (str): Person identifier ('sam' or 'jack')
#         identifier (str): Dataset identifier ('p1', 'p2', 'p3', 'OG')
    
#     Returns:
#         tuple: (exclude_indexes, exclude_datasets)
#     """
#     # Configuration dictionary mapping (person, identifier) to exclude configs
#     config = {
#         ### SAM
#         ('sam', 'OG'): {
#             'exclude_indexes': {
#                 'Child': [31], 
#                 'Joyce': [1, 30],
#                 'Justice': [1, 2, 3],
#                 'Question': [1, 17, 31],
#                 'Thought': [1, 29],
#                 'Through': [32],
#                 'Weapons': [32],
#                 'XRay2': [31],
#             },
#             'exclude_datasets': ['NOISE', 'NOISE2', 'NOISE3', 'XRay', 'Europe']
#         },
#         ('sam', 'p3'): {
#             'exclude_indexes': {
#                 'Justice': [14, 16, 27],
#                 'Through': [1, 20, 28, 31], 
#                 'Weapons': [31],
#             },
#             'exclude_datasets': []
#         },
#         ('sam', 'p2'): {
#             'exclude_indexes': {
#                 'Justice': [3, 22, 23, 32],
#                 'Through': [1, 18, 19, 30],
#                 'Weapons': [18, 20],
#             },
#             'exclude_datasets': []
#         },
#         ('sam', 'p1'): {
#             'exclude_indexes': {
#                 'Justice': [1, 8],
#                 'Weapons': [22, 29],
#             },
#             'exclude_datasets': []
#         },
        
#         ### JACK 
#         ('jack', 'p2'): {
#             'exclude_indexes': {
#                 'Child': [1], 
#                 'Europe': [],
#                 'Exactly': [30, 31],
#                 'Joyce': [31],
#                 'Justice': [31],
#                 'Question': [],
#                 'Thought': [31],
#                 'Through': [1],
#                 'Weapons': [1],
#                 'XRay2': [31],
#             },
#             'exclude_datasets': []
#         },
#         ### Alex
#         ('alex', 'p1'): {
#             'exclude_indexes': {
#                 'aid': [6],
#                 'key': [1],
#                 'my': [2],
#                 'same': [2],
#                 'seam': [1], 
#             },
#             'exclude_datasets': ['key', 'seam']
#         },
#         ('alex', 'p3'): {
#             'exclude_indexes': {
#                 'day': [7, 15],
#                 'dude': [29, 31],
#                 'my': [27],
#                 'same': [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60], 
#             },
#             'exclude_datasets': []
#         },
#     }
    
#     # Convert inputs to lowercase for case-insensitive matching
#     person = person.lower()
#     identifier = identifier.lower()
    
#     # Look up the configuration
#     key = (person, identifier)
#     if key in config:
#         return config[key]['exclude_indexes'], config[key]['exclude_datasets']
#     else:
#         print(f"Warning: No configuration found for {person} and {identifier}. Using empty defaults.")
#         return {}, ['NOISE', 'NOISE2', 'NOISE3', 'XRay', 'Europe']

# # Usage example:
# exclude_indexes, exclude_datasets = get_exclude_config(PERSON, IDENTIFIER)

def exclude_from_json(dataset_path, dataset_index):
    """
    Loads exclusion configuration from a JSON file corresponding to a dataset .h5 file,
    applies it to the provided dataset_index dict, and returns the filtered version.

    Args:
        dataset_path (str): Path to the .h5 dataset file
                            e.g. '../data/p3_Alex_Recording.h5'
        dataset_index (dict): Mapping of dataset_name -> list of (start, end) segments

    Returns:
        dict: Filtered dataset_index after exclusions
    """
    # Derive JSON path
    base, _ = os.path.splitext(dataset_path)
    json_path = base + "_excluded.json"

    # Default empty exclusions
    exclude_indexes = {}
    exclude_datasets = []

    # Load JSON if available
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        exclude_indexes = config.get("exclude_indexes", {})
        exclude_datasets = config.get("exclude_datasets", [])
        print(f"Loaded exclusions from {os.path.basename(json_path)}")
    else:
        print(f"⚠️  No exclusion file found for {os.path.basename(dataset_path)}. Using defaults.")

    # Apply exclusions
    filtered_segments = {}
    for dataset, segments in dataset_index.items():
        if dataset in exclude_datasets:
            print(f"Excluding entire dataset: {dataset}")
            continue

        old_size = len(segments)
        filtered_segments[dataset] = [
            seg for i, seg in enumerate(segments)
            if (i + 1) not in exclude_indexes.get(dataset, [])
        ]
        new_size = len(filtered_segments[dataset])

        if new_size != old_size:
            print(f"Dataset: {dataset}, Old size: {old_size}, New size: {new_size}")
        else:
            print(f"Dataset: {dataset}, no changes")

    return filtered_segments


def plot_dataset(dataset, segments, loader, AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE, threshold_value=None, frame_times=None, ste=None, show_plots=True):
    """
    Plot waveform, short-time energy, and ADC data for a dataset.
    
    Parameters:
    -----------
    dataset : str
        Name of the dataset to plot
    segments : list
        List of (start_sample, end_sample) tuples for each detected segment
    loader : object
        Dataset loader object with load_dataset method
    AUDIO_SAMPLE_RATE : float
        Sample rate of audio data
    ADC_SAMPLE_RATE : float
        Sample rate of ADC data
    threshold_value : float, optional
        Threshold value for short-time energy plot
    frame_times : array-like, optional
        Time values for short-time energy frames
    ste : array-like, optional
        Short-time energy values
    show_plots : bool, default=True
        If False, the function will set up the plots but not display them,
        allowing for additional modifications or saving
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("Dataset:", dataset)
    data = loader.load_dataset(dataset)
    audio_data = data["audio_data"]
    adc_data = data["adc_data"]
    N_audio_samples = len(audio_data)
    N_adc_samples = len(adc_data[1]) + len(adc_data[3])
    
    # Calculate average ADC values if needed
    avg_adc_ch1 = np.average(adc_data[1])
    avg_adc_ch3 = np.average(adc_data[3])
    
    fig = plt.figure(figsize=(20, 10))
    
    # Audio Waveform Plot
    plt.subplot(3, 1, 1)
    times = np.arange(len(audio_data)) / AUDIO_SAMPLE_RATE
    plt.plot(times, audio_data, label="Waveform")
    y_text = max(audio_data)
    
    # Overlay detected segments as colored spans
    for i, (start_samp, end_samp) in enumerate(segments):
        plt.axvspan(start_samp/AUDIO_SAMPLE_RATE, end_samp/AUDIO_SAMPLE_RATE, color='green', alpha=0.3)
        plt.text((start_samp + end_samp) / (2 * AUDIO_SAMPLE_RATE), y_text, f"{i+1}", 
                 fontsize=8, color='black', ha='center', va='center')
    
    plt.title(f"Waveform for '{dataset}'")
    plt.xlabel("Time (sec)")
    plt.ylabel("Amplitude")
    plt.xlim([0, max(times)])
    plt.legend()
    plt.grid()
    
    # Short-Time Energy Plot (if provided)
    if frame_times is not None and ste is not None:
        plt.subplot(3, 1, 2)
        plt.plot(frame_times, ste, label="Short-Time Energy")
        
        if threshold_value is not None:
            plt.axhline(threshold_value, linestyle='--', label=f"Threshold={threshold_value:.4f}")
            
        plt.title(f"Short-Time Energy for '{dataset}'")
        plt.xlabel("Time (sec)")
        plt.ylabel("Energy")
        plt.legend()
        plt.grid()
    
    # ADC Data Plot
    plt.subplot(3, 1, 3)
    times = np.arange(len(adc_data[1])) / ADC_SAMPLE_RATE
    y_text = max(adc_data[1])
    
    print(f"ADC channels: {len(adc_data[1])}, {len(adc_data[3])}")
    
    plt.plot(times, adc_data[1] - avg_adc_ch1, label="ADC Data 1")
    plt.plot(times, adc_data[3] - avg_adc_ch3, label="ADC Data 3")
    
    for i, (start_samp, end_samp) in enumerate(segments):
        plt.axvspan(start_samp/(ADC_SAMPLE_RATE*12), end_samp/(ADC_SAMPLE_RATE*12), color='green', alpha=0.3)
        plt.text((start_samp + end_samp) / (2*(ADC_SAMPLE_RATE*12)), y_text, f"{i+1}", 
                 fontsize=8, color='black', ha='center', va='center')
                 
    plt.xlabel("Sample Number")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.title(f"ADC Data for '{dataset}'")
    
    plt.tight_layout()
    
    if show_plots:
        plt.show()
    
    return fig

# Example usage:
# # To plot a single dataset:
# plot_dataset('Through', filtered_segments['Through'], loader, AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE, 
#              threshold_value, frame_times, ste)

# To plot all datasets:
# def plot_all_datasets(filtered_segments, loader, AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE, 
#                       threshold_value=None, frame_times=None, ste=None, show_plots=True):
#     """
#     Plot all datasets in filtered_segments.

#     Parameters:
#     -----------
#     filtered_segments : dict
#         Dictionary mapping dataset names to their segment lists
#     loader : object
#         Dataset loader object with load_dataset() method
#     AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE : float
#         Sample rates for audio and ADC
#     threshold_value, frame_times, ste : optional
#         If provided globally, will be overridden by per-dataset STE computed from audio_data
#     show_plots : bool
#         Whether to display the plots
#     """
#     import numpy as np
#     figures = {}

#     for dataset, segments in filtered_segments.items():
#         print("Dataset:", dataset)
#         data = loader.load_dataset(dataset)
#         audio_data = data["audio_data"]

#         # 🔄 Always recompute STE per dataset (fixes identical-plot issue)
#         segments_local, ste_local, frame_times_local, threshold_local = short_time_energy_segmentation(
#             audio_data, AUDIO_SAMPLE_RATE,
#             frame_duration=0.25,
#             hop_duration=0.1,
#             smoothing_window=4,
#             energy_quantile=0.3,
#             min_silence_frames=1,
#             min_voiced_frames=3
#         )

#         # You can either use filtered_segments’ segments or the recomputed ones
#         # If you want to use your existing segmentation results:
#         segs_to_plot = segments
#         # If you want to override with recomputed segmentation:
#         # segs_to_plot = segments_local

#         figures[dataset] = plot_dataset(
#             dataset, segs_to_plot, loader, AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE,
#             threshold_local, frame_times_local, ste_local, show_plots
#         )

#     return figures


def plot_all_datasets(filtered_segments, loader, AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE, 
                      results=None, show_plots=True):
    """
    Plot all datasets using previously computed STE results when available.

    Parameters:
    -----------
    filtered_segments : dict
        Dictionary mapping dataset names to their segment lists.
    loader : object
        Dataset loader object with load_dataset() method.
    AUDIO_SAMPLE_RATE, ADC_SAMPLE_RATE : float
        Sample rates for audio and ADC.
    results : dict, optional
        Output from compute_and_plot_ste(). If provided, STE, frame_times,
        and threshold_value will be taken from here.
    show_plots : bool
        Whether to display the plots.

    Returns:
    --------
    dict
        Mapping dataset names → matplotlib figure handles.
    """
    import numpy as np
    figures = {}

    for dataset, segments in filtered_segments.items():
        #print("Dataset:", dataset)
        data = loader.load_dataset(dataset)
        audio_data = data["audio_data"]

        # If we already have precomputed STE data, use it
        if results is not None and dataset in results:
            ste_local = results[dataset]["ste"]
            frame_times_local = results[dataset]["frame_times"]
            threshold_local = results[dataset]["threshold_value"]
        else:
            # Fallback: compute if not provided
            print(f"⚠️  No precomputed STE found for '{dataset}', recomputing...")
            segments_local, ste_local, frame_times_local, threshold_local = short_time_energy_segmentation(
                audio_data, AUDIO_SAMPLE_RATE,
                frame_duration=0.25,
                hop_duration=0.1,
                smoothing_window=4,
                energy_quantile=0.3,
                min_silence_frames=1,
                min_voiced_frames=3
            )

        # Choose which segments to plot (filtered vs recomputed)
        segs_to_plot = segments

        figures[dataset] = plot_dataset(
            dataset,
            segs_to_plot,
            loader,
            AUDIO_SAMPLE_RATE,
            ADC_SAMPLE_RATE,
            threshold_local,
            frame_times_local,
            ste_local,
            show_plots
        )

    return figures

def BPfilter(data, fs, lowcut_hz=None, highcut_hz=None):
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
    # Default cutoff frequencies if not provided
    if lowcut_hz is None:
        lowcut_hz = 20  # Default lower cutoff of 20 Hz
    
    if highcut_hz is None:
        highcut_hz = fs/4  # Default upper cutoff at quarter of sampling rate
    
    # Convert cutoff frequencies to normalized units (0 to 1)
    nyquist = fs / 2
    low = lowcut_hz / nyquist
    high = highcut_hz / nyquist
    
    # Create a 4th-order bandpass Butterworth filter
    b, a = signal.butter(2, [low, high], btype='band')
    
    # Apply zero-phase filtering using filtfilt
    filtered_data = signal.filtfilt(b, a, data)
    # filtered_data = data
    
    return filtered_data


def filter_signals(filtered_segments, loader,
                   audio_sampling_rate=48000,
                   adc_sampling_rate=8000,
                   audio_lowcut=300,
                   audio_highcut=4000,
                   adc_lowcut=5,
                   adc_highcut=3700):
    """
    Extract and bandpass-filter audio and ADC segments from datasets.

    Args:
        filtered_segments (dict): Mapping dataset_name → list of (start, end) sample pairs.
        loader (H5DataLoader): Loader with load_dataset() method.
        audio_sampling_rate (float): Sampling rate for audio channel.
        adc_sampling_rate (float): Sampling rate for ADC channels.
        audio_lowcut, audio_highcut (float): Audio bandpass cutoff frequencies (Hz).
        adc_lowcut, adc_highcut (float): ADC bandpass cutoff frequencies (Hz).

    Returns:
        tuple: (audiosegmentsX, adc1segmentsX, adc2segmentsX, segmentsY,
                min_lengths, max_lengths, y_minmax)
            where:
                - audiosegmentsX, adc1segmentsX, adc2segmentsX : list of np.arrays
                - segmentsY : list of dataset labels
                - min_lengths, max_lengths : dicts with min/max lengths
                - y_minmax : dict with overall min/max amplitudes
    """
    audiosegmentsX = []
    adc1segmentsX = []
    adc2segmentsX = []
    segmentsY = []

    # --- Extract and filter segments ---
    for dataset, segments in filtered_segments.items():
        print(f"Processing dataset: {dataset}")
        data = loader.load_dataset(dataset)
        audio_data = data["audio_data"]
        adc_data = data["adc_data"]

        for start_samp, end_samp in segments:
            # Extract blocks
            audioblock = audio_data[start_samp:end_samp]
            adc1block = adc_data[1][start_samp // 12:end_samp // 12]
            adc2block = adc_data[3][start_samp // 12:end_samp // 12]

            # Apply bandpass filters
            audio_filtered = BPfilter(audioblock, audio_sampling_rate, audio_lowcut, audio_highcut)
            adc1_filtered = BPfilter(adc1block, adc_sampling_rate, adc_lowcut, adc_highcut)
            adc2_filtered = BPfilter(adc2block, adc_sampling_rate, adc_lowcut, adc_highcut)

            # Store results
            audiosegmentsX.append(audio_filtered)
            adc1segmentsX.append(adc1_filtered)
            adc2segmentsX.append(adc2_filtered)
            segmentsY.append(dataset)

    # --- Compute length stats ---
    min_lengths = {
        "adc1": min(len(x) for x in adc1segmentsX),
        "adc2": min(len(x) for x in adc2segmentsX),
        "audio": min(len(x) for x in audiosegmentsX),
    }
    max_lengths = {
        "adc1": max(len(x) for x in adc1segmentsX),
        "adc2": max(len(x) for x in adc2segmentsX),
        "audio": max(len(x) for x in audiosegmentsX),
    }

    print(f"Min lengths: {min_lengths}")
    print(f"Max lengths: {max_lengths}")

    # --- Compute amplitude stats ---
    audio_y_min = min(np.min(x) for x in audiosegmentsX)
    audio_y_max = max(np.max(x) for x in audiosegmentsX)
    adc_y_min = min(
        min(np.min(x) for x in adc1segmentsX),
        min(np.min(x) for x in adc2segmentsX)
    )
    adc_y_max = max(
        max(np.max(x) for x in adc1segmentsX),
        max(np.max(x) for x in adc2segmentsX)
    )

    y_minmax = {
        "audio": (audio_y_min, audio_y_max),
        "adc": (adc_y_min, adc_y_max)
    }

    print(f"Audio amplitude range: {audio_y_min:.4f} to {audio_y_max:.4f}")
    print(f"ADC amplitude range: {adc_y_min:.4f} to {adc_y_max:.4f}")

    return (
        audiosegmentsX,
        adc1segmentsX,
        adc2segmentsX,
        segmentsY,
        min_lengths,
        max_lengths,
        y_minmax
    )

def process_segments(
    segments,
    identifier,
    person,
    loader,
    filtered=True,
    test=False,
    audio_sampling_rate=48000,
    adc_sampling_rate=8000,
    audio_lowcut=300,
    audio_highcut=4000,
    adc_lowcut=5,
    adc_highcut=None,
):
    """
    Create a memmap of segments and a descriptor JSON.

    IMPORTANT: this version does *no* bandpass filtering. It stores RAW
    audio/adc segments with np.inf padding. Filtering is done later in
    MemmapDataset(__getitem__) when you pass filter=True.

    Parameters
    ----------
    segments : dict
        { dataset_name : [ (start_samp, end_samp), ... ] }
        start/end are in *audio* sample indices.
    identifier : str
        e.g. 'p6'
    person : str
        e.g. 'alex'
    loader : H5DataLoader-like
        loader.load_dataset(name) -> { "audio_data": array,
                                       "adc_data": {1: ch1, 3: ch3} }
    filtered : bool
        Only used for file naming:
        - True  -> "..._filtered_..."
        - False -> "..._noise_..."
    test : bool
        Not implemented here. Use test=False with MemmapDataset
        (since it expects 'n_segments' in descriptor).
    """

    if test:
        raise NotImplementedError("Use test=False for MemmapDataset compatibility.")

    if adc_highcut is None:
        adc_highcut = adc_sampling_rate / 4.0

    # ------------------------------------------------------------------
    # Compute max lengths and total number of segments
    # ------------------------------------------------------------------
    max_audio_len = max(
        end_samp - start_samp
        for segs in segments.values()
        for start_samp, end_samp in segs
    )
    max_adc_len = max_audio_len // 12  # your usual 48k / 4k ratio
    n_segments = sum(len(segs) for segs in segments.values())

    # ------------------------------------------------------------------
    # Prepare memmap
    # ------------------------------------------------------------------
    suffix = "filtered" if filtered else "noise"
    memmap_filename = f"{identifier}{person}_{suffix}.dat"

    if os.path.exists(memmap_filename):
        os.remove(memmap_filename)

    dtype = np.dtype([
        ('id',    np.int32),
        ('audio', np.float32, (max_audio_len,)),
        ('adc1',  np.float32, (max_adc_len,)),
        ('adc2',  np.float32, (max_adc_len,))
    ])

    mm = np.memmap(memmap_filename, dtype=dtype, mode='w+', shape=(n_segments,))

    # For stats
    all_audio_values = []
    all_adc_values   = []

    id2dataset = {}
    currentid = -1
    iterator = 0

    for dataset, segs in segments.items():
        currentid += 1
        id2dataset[currentid] = dataset

        data = loader.load_dataset(dataset)
        audio_data = data["audio_data"]
        adc_data   = data["adc_data"]

        for start_samp, end_samp in segs:
            # Raw blocks (no filtering here!)
            audioblock = audio_data[start_samp:end_samp]
            adc1block  = adc_data[1][start_samp // 12 : end_samp // 12]
            adc2block  = adc_data[3][start_samp // 12 : end_samp // 12]

            # Accumulate for stats
            all_audio_values.append(audioblock)
            all_adc_values.append(adc1block)
            all_adc_values.append(adc2block)

            # Write scalar ID
            mm['id'][iterator] = currentid

            # Pad audio with +inf
            padded_audio = np.ones(max_audio_len, dtype=np.float32) * np.inf
            padded_audio[:len(audioblock)] = audioblock
            mm['audio'][iterator] = padded_audio

            # Pad ADC1
            padded_adc1 = np.ones(max_adc_len, dtype=np.float32) * np.inf
            padded_adc1[:len(adc1block)] = adc1block
            mm['adc1'][iterator] = padded_adc1

            # Pad ADC2
            padded_adc2 = np.ones(max_adc_len, dtype=np.float32) * np.inf
            padded_adc2[:len(adc2block)] = adc2block
            mm['adc2'][iterator] = padded_adc2

            iterator += 1

    mm.flush()

    # ------------------------------------------------------------------
    # Compute global stats (ignoring padding; we used raw blocks)
    # ------------------------------------------------------------------
    all_audio_values = np.concatenate(all_audio_values).astype(np.float32)
    all_adc_values   = np.concatenate(all_adc_values).astype(np.float32)

    audio_mean = float(np.mean(all_audio_values))
    audio_std  = float(np.std(all_audio_values))
    adc_mean   = float(np.mean(all_adc_values))
    adc_std    = float(np.std(all_adc_values))

    # ------------------------------------------------------------------
    # Descriptor JSON
    # ------------------------------------------------------------------
    descriptor_dict = {
        'audio_sampling_rate': audio_sampling_rate,
        'adc_sampling_rate':   adc_sampling_rate,
        'audio_lowcut':        audio_lowcut,
        'audio_highcut':       audio_highcut,
        'adc_lowcut':          adc_lowcut,
        'adc_highcut':         adc_highcut,
        'max_audio_len':       int(max_audio_len),
        'max_adc_len':         int(max_adc_len),
        'n_segments':          int(n_segments),
        'memmap_filename':     memmap_filename,
        'dataset_mapping':     id2dataset,   # {id: dataset_name}
        'dtype':               dtype.descr,
        # NEW: needed by your normalizer
        'audio_mean':          audio_mean,
        'audio_std':           audio_std,
        'adc_mean':            adc_mean,
        'adc_std':             adc_std,
    }

    descriptor_filename = f"{identifier}{person}_{suffix}_descriptor.json"
    with open(descriptor_filename, 'w') as f:
        json.dump(descriptor_dict, f)

    mapping_filename = f"{identifier}{person}_{suffix}_id2dataset.json"
    with open(mapping_filename, 'w') as f:
        json.dump(id2dataset, f)

    print(f"Data saved to {memmap_filename}")
    print(f"Descriptor saved to {descriptor_filename}")
    print(f"Mapping saved to {mapping_filename}")

    # This function’s *main* purpose is disk export;
    # if you need in-RAM arrays, you can return None or adapt.
    return None, None, None, None

def plot_signal_spectrograms(audiosegmentsX, adc1segmentsX, adc2segmentsX, segmentsY, 
                             adc_sampling_rate, use_mel=False, segments_per_word=3, 
                             freq_range=None, color_by_word=True, use_clahe=False, 
                             clahe_clip_limit=2.0, clahe_tile_grid_size=(8,8), interpolation='gaussian'):
    """
    Plot raw signals and spectrograms (either regular or mel) for audio and ADC data.
    
    Parameters:
    -----------
    audiosegmentsX : list
        List of audio segments
    adc1segmentsX : list
        List of ADC1 segments
    adc2segmentsX : list
        List of ADC2 segments
    segmentsY : list
        Labels for the segments
    adc_sampling_rate : int
        Sampling rate in Hz
    use_mel : bool, optional
        If True, use mel spectrograms, otherwise use regular spectrograms (default: False)
    segments_per_word : int, optional
        Number of segments to plot per unique word (default: 3)
    freq_range : tuple, optional
        Frequency range (min, max) to display for ADC spectrograms (default: None)
    color_by_word : bool, optional
        If True, use different colors for each unique word (default: True)
    interpolation : str, optional
        Interpolation method for spectrograms ('gaussian', 'bicubic', 'bilinear', etc.) (default: 'gaussian')
    
    Returns:
    --------
    None
    """
    # First, identify the unique words in segmentsY
    unique_words = []
    word_segment_counts = {}
    
    # Count how many segments we have for each word
    for word in segmentsY:
        if word not in word_segment_counts:
            unique_words.append(word)
            word_segment_counts[word] = 0
        word_segment_counts[word] += 1
    
    # For each word, take up to segments_per_word segments
    segments_to_plot = []
    for word in unique_words:
        count = 0
        for i, segment_word in enumerate(segmentsY):
            if segment_word == word and count < segments_per_word:
                segments_to_plot.append(i)
                count += 1
    
    # Sort the indices to maintain the original order
    segments_to_plot.sort()
    
    # Create a color map for unique words if color_by_word is True
    word_colors = {}
    if color_by_word:
        colormap = cm.get_cmap('tab10')  # Use a colormap with distinct colors
        norm = Normalize(vmin=0, vmax=len(unique_words)-1)
        for i, word in enumerate(unique_words):
            word_colors[word] = colormap(norm(i))
    
    # Create figure
    plt.figure(figsize=(18, 6 * len(segments_to_plot)))
    
    # Spectrogram parameters
    if use_mel:
        # Mel spectrogram parameters
        # Audio parameters
        audio_n_fft = 2048
        audio_hop_length = 512
        audio_n_mels = 128
        
        # ADC parameters (smaller window for better time resolution)
        adc_n_fft = 1024
        adc_hop_length = 256
        adc_n_mels = 64
    else:
        # Regular spectrogram parameters
        nfft = 512
        noverlap = 384
    
    # Plot each selected segment
    for plot_idx, i in enumerate(segments_to_plot):
        # Get the current segments
        audioblock = audiosegmentsX[i]
        adc1block = adc1segmentsX[i]
        adc2block = adc2segmentsX[i]

        # Base index for this segment's subplot grid (6 plots per segment)
        base_idx = plot_idx * 6 + 1
        
        # Get current word and its color
        current_word = segmentsY[i]
        
        # Determine colors based on word
        if color_by_word:
            line_color = word_colors[current_word]
            title_color = word_colors[current_word]
        else:
            line_color = 'b'  # Default blue
            title_color = 'black'
        
        # Row 1: Raw data (time domain)
        # Audio raw data
        plt.subplot(2 * len(segments_to_plot), 3, base_idx)
        plt.plot(audioblock, color=line_color)
        plt.title(f"Audio Segment {i+1} - {current_word}", color=title_color)
        
        # ADC1 raw data
        plt.subplot(2 * len(segments_to_plot), 3, base_idx + 1)
        plt.plot(adc1block, color=line_color)
        plt.title(f"ADC1 Segment {i+1} - {current_word}", color=title_color)
        
        # ADC2 raw data
        plt.subplot(2 * len(segments_to_plot), 3, base_idx + 2)
        plt.plot(adc2block, color=line_color)
        plt.title(f"ADC2 Segment {i+1} - {current_word}", color=title_color)
        
        # Row 2: Spectrograms
        if use_mel:
            # MEL SPECTROGRAMS
            
            # Audio mel spectrogram
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 3)
            mel_spec_audio = librosa.feature.melspectrogram(
                y=audioblock.astype(float), 
                sr=adc_sampling_rate,
                n_fft=audio_n_fft,
                hop_length=audio_hop_length,
                n_mels=audio_n_mels
            )
            mel_spec_db_audio = librosa.power_to_db(mel_spec_audio, ref=np.max)
            
            # Use librosa's specshow for the axes and labels, but don't show the image
            librosa.display.specshow(
                mel_spec_db_audio, 
                sr=adc_sampling_rate, 
                hop_length=audio_hop_length,
                x_axis='time', 
                y_axis='mel', 
                cmap='viridis',
                alpha=0  # Make this transparent
            )
            
            # Then overlay with imshow which supports interpolation
            plt.imshow(mel_spec_db_audio, aspect='auto', origin='lower', 
                      interpolation=interpolation, cmap='viridis')
                        
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"Audio Mel Spectrogram {i+1} - {current_word}", color=title_color)
            
            # ADC1 mel spectrogram
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 4)
            mel_spec_adc1 = librosa.feature.melspectrogram(
                y=adc1block.astype(float), 
                sr=adc_sampling_rate,
                n_fft=adc_n_fft,
                hop_length=adc_hop_length,
                n_mels=adc_n_mels
            )
            mel_spec_db_adc1 = librosa.power_to_db(mel_spec_adc1, ref=np.max)
            
            # Use librosa's specshow for the axes and labels, but don't show the image
            librosa.display.specshow(
                mel_spec_db_adc1, 
                sr=adc_sampling_rate, 
                hop_length=adc_hop_length,
                x_axis='time', 
                y_axis='mel', 
                cmap='viridis',
                alpha=0  # Make this transparent
            )
            
            # Then overlay with imshow which supports interpolation
            plt.imshow(mel_spec_db_adc1, aspect='auto', origin='lower', 
                      interpolation=interpolation, cmap='viridis')
            
            plt.colorbar(format='%+2.0f dB')
            title_suffix = ""
            if freq_range:
                plt.ylim(freq_range)
                title_suffix = f" ({freq_range[0]}-{freq_range[1]} Hz)"
            plt.title(f"ADC1 Mel Spectrogram {i+1} - {current_word}{title_suffix}", color=title_color)
            
            # ADC2 mel spectrogram
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 5)
            mel_spec_adc2 = librosa.feature.melspectrogram(
                y=adc2block.astype(float), 
                sr=adc_sampling_rate,
                n_fft=adc_n_fft,
                hop_length=adc_hop_length,
                n_mels=adc_n_mels
            )
            mel_spec_db_adc2 = librosa.power_to_db(mel_spec_adc2, ref=np.max)
            
            # Use librosa's specshow for the axes and labels, but don't show the image
            librosa.display.specshow(
                mel_spec_db_adc2, 
                sr=adc_sampling_rate, 
                hop_length=adc_hop_length,
                x_axis='time', 
                y_axis='mel', 
                cmap='viridis',
                alpha=0  # Make this transparent
            )
            
            # Then overlay with imshow which supports interpolation
            plt.imshow(mel_spec_db_adc2, aspect='auto', origin='lower', 
                      interpolation=interpolation, cmap='viridis')
            
            plt.colorbar(format='%+2.0f dB')
            if freq_range:
                plt.ylim(freq_range)
            plt.title(f"ADC2 Mel Spectrogram {i+1} - {current_word}{title_suffix}", color=title_color)
            
        else:
            # REGULAR SPECTROGRAMS
            
            # Audio spectrogram - full frequency range
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 3)
            plt.specgram(audioblock, NFFT=nfft, Fs=adc_sampling_rate, noverlap=noverlap, 
                         scale='dB', cmap='viridis')
            plt.title(f"Audio Spectrogram {i+1} - {current_word}", color=title_color)
            plt.ylabel('Frequency [Hz]')
            
            # ADC1 spectrogram
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 4)
            plt.specgram(adc1block, NFFT=nfft, Fs=adc_sampling_rate, noverlap=noverlap, 
                         scale='dB', cmap='viridis')
            title_suffix = ""
            if freq_range:
                plt.ylim(freq_range)
                title_suffix = f" ({freq_range[0]}-{freq_range[1]} Hz)"
            plt.title(f"ADC1 Spectrogram {i+1} - {current_word}{title_suffix}", color=title_color)
            plt.ylabel('Frequency [Hz]')
            
            # ADC2 spectrogram
            plt.subplot(2 * len(segments_to_plot), 3, base_idx + 5)
            plt.specgram(adc2block, NFFT=nfft, Fs=adc_sampling_rate, noverlap=noverlap, 
                         scale='dB', cmap='viridis')
            if freq_range:
                plt.ylim(freq_range)
            plt.title(f"ADC2 Spectrogram {i+1} - {current_word}{title_suffix}", color=title_color)
            plt.ylabel('Frequency [Hz]')
    
    # Add a legend showing the words and their corresponding colors
    if color_by_word:
        # Create a small legend plot at the top of the figure
        plt.figtext(0.5, 0.98, "Word Color Legend", ha="center", fontsize=12, weight='bold')
        legend_handles = []
        for i, word in enumerate(unique_words):
            legend_handles.append(plt.Line2D([0], [0], color=word_colors[word], lw=4, label=word))
        plt.figlegend(handles=legend_handles, loc='upper center', 
                     bbox_to_anchor=(0.5, 0.96), ncol=min(5, len(unique_words)))
            
    plt.tight_layout()
    plt.show()


# Function for speech feature extraction with only mel spectrogram features
def extract_speech_features(audio_segments, sr=8000):
    # Use your specified parameters
    audio_n_fft = 1024
    audio_hop_length = 32
    audio_n_mels = 32
    
    features = []
    
    for audio in audio_segments:
        # Extract mel spectrogram using your parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_fft=audio_n_fft,
            hop_length=audio_hop_length,
            n_mels=audio_n_mels,
            fmin=6,     # Lower bound for speech
            fmax=8000   # Upper frequency limit for speech
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec)
        
        # Statistical aggregation of mel spectrogram
        mel_mean = np.mean(log_mel_spec, axis=1)
        mel_std = np.std(log_mel_spec, axis=1)
        
        # Combine features
        speech_features = np.concatenate([
            mel_mean, mel_std
        ])
        
        features.append(speech_features)
    
    return np.array(features)

def analyze_feature_clustering(
    X_combined,
    y,
    test_size=0.5,
    n_neighbors=15,
    min_dist=0.1,
    metric="correlation",
    random_state=42,
    plot=True,
    save_path=None
):
    """
    Perform label encoding, scaling, UMAP dimensionality reduction, and K-means clustering,
    then visualize and compute clustering metrics.

    Args:
        X_combined (array-like): Feature matrix.
        y (array-like): Label vector.
        test_size (float): Fraction for test split.
        n_neighbors (int): Number of neighbors for UMAP.
        min_dist (float): Minimum distance for UMAP embedding.
        metric (str): Metric for UMAP (e.g., 'correlation', 'euclidean').
        random_state (int): Random seed for reproducibility.
        plot (bool): Whether to display the UMAP + clustering plots.
        save_path (str): Optional path to save figure (e.g., 'clustering.png').

    Returns:
        dict: {
            "X_umap": np.ndarray,
            "clusters": np.ndarray,
            "y_encoded": np.ndarray,
            "label_encoder": LabelEncoder,
            "metrics": {"ARI": float, "NMI": float}
        }
    """
    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)

    # --- UMAP dimensionality reduction ---
    umap_combined = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    X_umap = umap_combined.fit_transform(X_scaled)

    # --- K-means clustering ---
    num_classes = len(np.unique(y_test_encoded))
    kmeans = KMeans(n_clusters=num_classes, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # --- Compute metrics ---
    ari = adjusted_rand_score(y_test_encoded, clusters)
    nmi = normalized_mutual_info_score(y_test_encoded, clusters)
    print(f"Adjusted Rand Index (ARI): {ari:.3f}")
    print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

    # --- Visualization ---
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))

        # Left: Original labels
        scatter1 = axes[0].scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=y_test_encoded, cmap="tab20", edgecolor="k", s=80, alpha=0.7
        )
        axes[0].set_title("UMAP Projection - Original Labels", fontsize=16)
        axes[0].set_xlabel("UMAP Component 1")
        axes[0].set_ylabel("UMAP Component 2")
        axes[0].grid(alpha=0.3)

        handles, _ = scatter1.legend_elements()
        labels = label_encoder.inverse_transform(np.unique(y_test_encoded))
        axes[0].legend(handles, labels, title="True Labels", loc="best", fontsize=12)

        # Right: Clustered labels
        scatter2 = axes[1].scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=clusters, cmap="tab20", edgecolor="k", s=80, alpha=0.7
        )
        axes[1].set_title(f"K-means Clusters (k={num_classes})", fontsize=16)
        axes[1].set_xlabel("UMAP Component 1")
        axes[1].set_ylabel("UMAP Component 2")
        axes[1].grid(alpha=0.3)

        handles, _ = scatter2.legend_elements()
        axes[1].legend(
            handles, [f"Cluster {i+1}" for i in range(len(handles))],
            title="Discovered Clusters", loc="best", fontsize=12
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Figure saved to {save_path}")
        plt.show()

    # --- Return results ---
    return {
        "X_umap": X_umap,
        "clusters": clusters,
        "y_encoded": y_test_encoded,
        "label_encoder": label_encoder,
        "metrics": {"ARI": ari, "NMI": nmi}
    }