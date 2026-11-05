import numpy as np
from skimage.filters import threshold_otsu
import random
from scipy import signal
import sys
import mne

def bandpass_fir_mne(
    x,
    fs,
    f_lo,
    f_hi,
    filter_length="0.5s",
    l_trans_bandwidth=10.0,
    h_trans_bandwidth=10.0,
):
    """
    Zero-phase band-pass filter via MNE.

    Input x must have time on the last dimension, e.g. (n_channels, n_times).

    For low cutoffs (e.g. 4-30 Hz) on 0.5 s data, FIR would need a longer filter
    than the segment. So when f_lo < 8 Hz we use IIR instead (short effective
    length, 4-30 Hz works on 0.5 s). MNE applies IIR via filtfilt (zero-phase);
    order-4 Butterworth is standard and stable. Otherwise we use FIR.
    """
    x = x.astype(np.float64)
    n_samples = x.shape[-1]

    # IIR works on short segments and supports 4 Hz; use it when low cutoff is low
    if f_lo < 8:
        return mne.filter.filter_data(
            x,
            sfreq=fs,
            l_freq=f_lo,
            h_freq=f_hi,
            method="iir",
            iir_params={"order": 4, "ftype": "butter"},
            verbose=False,
        )

    # FIR for higher bands (e.g. 80-200 Hz)
    l_trans_bandwidth = min(l_trans_bandwidth, max(0.5, f_lo - 0.5))
    if n_samples < int(np.ceil(0.5 * fs)):
        filter_length = max(3, (n_samples // 2) * 2 + 1)
        l_trans_bandwidth = max(l_trans_bandwidth, 3.3 * fs / n_samples)
        l_trans_bandwidth = min(l_trans_bandwidth, max(0.5, f_lo - 0.5))
        h_trans_bandwidth = max(h_trans_bandwidth, 3.3 * fs / n_samples)

    return mne.filter.filter_data(
        x,
        sfreq=fs,
        l_freq=f_lo,
        h_freq=f_hi,
        method="fir",
        phase="zero",
        fir_design="firwin",
        pad="reflect_limited",
        filter_length=filter_length,
        l_trans_bandwidth=l_trans_bandwidth,
        h_trans_bandwidth=h_trans_bandwidth,
        verbose=False,
    )
    
def myisnan(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6

def ispadding(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6



def get_otsu_regions(out,nmm, labels, args_params = None):
    """ Identify DeepSIF source region using otsu_threshould, run on CPU
    :param out: np.arrry; the output of DeepSIF, batch_size * num_time * num_region
    :param labels: np.arrry; group truth source region; batch_size * num_source * max_size; starts from 0
    :param args_params: optional parameters, could be
                        dis_matrix: np.array; distance between regions; num_region (994) * num_region
    :return return_eval: could be
                         all_regions: DeepSIF predicted regions; (batch_size, )
                         all_out:     DeepSIF predicted source activity;  (batch_size, )
    """
    # when there is no spike, the location error is nan

    batch_size = labels.shape[0]
    return_eval = dict()

    return_eval['all_regions'] = np.empty((batch_size,), dtype=object)
    return_eval['all_out'] = np.empty((batch_size,), dtype=object)

    # Store per-source detected regions for metric calculation (used in HFO mode)
    per_source_detected = {}  # {example_idx: {source_idx: detected_regions}}
    per_source_gt = {}  # {example_idx: {source_idx: gt_regions}}
    
    if args_params['hfo']:
        
        for i in range(batch_size):
            # For HFO: process each source separately using time window around GT HFO
            # Extract number of sources
            num_sources = labels.shape[1]
            all_detected_regions = []
            all_gt_regions = []
            
            per_source_detected[i] = {}
            per_source_gt[i] = {}
            
            # Process each source separately
            for src_idx in range(num_sources):
                # Extract GT regions for this source (0-indexed)
                source_regions = labels[i, src_idx, :]
                source_regions = source_regions[np.logical_not(ispadding(source_regions))]
                source_regions = source_regions[source_regions >= 0]
                source_regions = source_regions[source_regions < out[i].shape[1]]  # Ensure within bounds
                source_regions = np.unique(source_regions)
                
                if len(source_regions) == 0:
                    continue  # Skip if no valid regions
                
                # Store GT regions for this source
                per_source_gt[i][src_idx] = source_regions
                
                # Find HFO time window: use peak time at center region
                center_region = source_regions[0]  # First region is center
                center_region_1idx = int(center_region)  # Already 0-indexed in Python
                
                if center_region_1idx >= 0 and center_region_1idx < nmm[i].shape[1]:
                    # Find peak time in NMM at center region
                    peak_time = np.argmax(np.abs(nmm[i][:, center_region_1idx]))
                    
                    # Define time window around HFO (±50 samples around peak)
                    window_half = 50  # Half window size
                    time_start = max(0, peak_time - window_half)
                    time_end = min(out[i].shape[0], peak_time + window_half + 1)
                    time_window = np.arange(time_start, time_end)
                    
                    # Calculate energy in this time window for each region
                    # Sum absolute values across time window
                    out_window = out[i][time_window, :]  # (window_size, regions)
                    thre_source = np.sum(np.abs(out_window), axis=0)  # (regions,)
                    
                    if np.max(thre_source) > 0:
                        thre_source = thre_source / np.max(thre_source)  # Normalize
                        # Apply Otsu threshold
                        thresh = threshold_otsu(thre_source)
                        # Get regions above threshold (0-indexed)
                        detected_regions = np.where(thre_source > thresh)[0]
                        per_source_detected[i][src_idx] = detected_regions
                        all_detected_regions.extend(detected_regions)
                        all_gt_regions.extend(source_regions)
                    else:
                        per_source_detected[i][src_idx] = np.array([], dtype=int)
                else:
                    per_source_detected[i][src_idx] = np.array([], dtype=int)
            
            # Combine all detected regions (union) for all_regions output
            all_detected_regions = np.unique(all_detected_regions)
            return_eval['all_regions'][i] = all_detected_regions
            if len(all_detected_regions) > 0:
                return_eval['all_out'][i] = out[i][:, all_detected_regions]
            else:
                return_eval['all_out'][i] = np.array([])
    else:
        for i in range(batch_size):
            
            for source in range(labels.shape[1]):
                peak_time = np.argmax(nmm[i, :, labels[i, :, 0]])

            thre_source = np.abs(out[i])
            # peak_magnitude = np.max(thre_source,axis= 0)
            # peak_region = np.argmax(peak_magnitude)
            # peak_time = np.argmax(thre_source[:,peak_region])
            thre_source = thre_source[peak_time,:]
            thre_source = (thre_source - np.min(thre_source)) / np.max(thre_source)
            thresh = threshold_otsu(thre_source)
            otsu_region = np.where(thre_source > thresh)[0]
            # thre_source = (thre_source - np.min(thre_source)) / np.max(thre_source)
            # thresh = threshold_otsu(thre_source, nbins=100)
            # select_pixel = thre_source > thresh
            # otsu_region = np.where(np.sum(select_pixel, axis=0) > 7)[0]
            return_eval['all_regions'][i] = otsu_region
            return_eval['all_out'][i] = out[i, :, otsu_region]

    # Calculate the eval metrics in Python
    if args_params is not None:
        return_eval['precision'] = np.zeros(batch_size)
        return_eval['recall'] = np.zeros(batch_size)
        return_eval['le'] = np.zeros(batch_size)
        return_eval['le_1'] = np.zeros(batch_size)
        return_eval['le_2'] = np.zeros(batch_size)
        
        for i in range(batch_size):
            if args_params['hfo']:
                # For HFO: calculate metrics per source, then average
                precision_per_source = []
                recall_per_source = []
                le_per_source = []
                
                if i in per_source_detected and i in per_source_gt:
                    for src_idx in per_source_gt[i].keys():
                        if src_idx not in per_source_detected[i]:
                            continue
                        
                        gt_regions_src = per_source_gt[i][src_idx]
                        detected_regions_src = per_source_detected[i][src_idx]
                        
                        if len(detected_regions_src) == 0:
                            continue  # Skip if no detected regions for this source
                        
                        if 'dis_matrix' in args_params and args_params['dis_matrix'] is not None:
                            # Calculate metrics for this source
                            overlap = len(np.intersect1d(gt_regions_src, detected_regions_src))
                            
                            if len(detected_regions_src) > 0:
                                precision_per_source.append(overlap / len(detected_regions_src))
                            if len(gt_regions_src) > 0:
                                recall_per_source.append(overlap / len(gt_regions_src))
                            
                            # Location Error
                            dist_submatrix = args_params['dis_matrix'][detected_regions_src][:, gt_regions_src]
                            le_per_source.append((np.mean(np.min(dist_submatrix, axis=0)) + np.mean(np.min(dist_submatrix, axis=1))) / 2)
                
                # Average per-source metrics
                if len(precision_per_source) > 0:
                    return_eval['precision'][i] = np.mean(precision_per_source)
                    return_eval['recall'][i] = np.mean(recall_per_source)
                    return_eval['le'][i] = np.mean(le_per_source)
                else:
                    return_eval['precision'][i] = None
                    return_eval['recall'][i] = None
                    return_eval['le'][i] = None
            else:
                # For non-HFO: combine all sources (original logic)
                # Combine all GT regions from all sources (union)
                lb = []
                for src_idx in range(labels.shape[1]):
                    source_regions = labels[i, src_idx, :]
                    source_regions = source_regions[np.logical_not(ispadding(source_regions))]
                    source_regions = source_regions[source_regions >= 0]
                    lb.extend(source_regions)
                lb = np.unique(lb)
                recon = return_eval['all_regions'][i]

                if len(recon) == 0:
                     return_eval['precision'][i] = None
                     return_eval['recall'][i] = None
                     return_eval['le'][i] = None
                else:
                    overlap_region = len(np.intersect1d(lb, recon))
                    # number of region based precision and recall
                    return_eval['precision'][i] = overlap_region/len(recon)
                    return_eval['recall'][i] = overlap_region /len(lb)
                    return_eval['le'][i] = (np.mean(np.min(args_params['dis_matrix'][recon][:, lb], axis = 0)) + np.mean(np.min(args_params['dis_matrix'][recon][:, lb], axis = 1)))/2
                    return_eval['le_1'][i] = np.mean(np.min(args_params['dis_matrix'][recon][:, lb], axis = 0))
                    return_eval['le_2'][i] = np.mean(np.min(args_params['dis_matrix'][recon][:, lb], axis = 1))
                    # return_eval['sd_v'][i] = np.min(args_params['dis_matrix'][recon][:, lb], axis = 1)
              

    return return_eval


def add_white_noise(sig, snr, args_params=None):
    """
    :param sig: np.array; num_electrode * num_time
    :param snr: int; signal to noise level in dB
    :param args_params: optional parameters, could be
                        ratio: np.array; ratio between white Gaussian noise and pre-set realistic noise
                        rndata: np.array; realistic noise data; num_sample * num_electrode * num_time
                        rnpower: np.array; pre-calculated power for rndata; num_sample * num_electrode

    :return: noise_sig: np.array; num_electrode * num_time
    """

    num_elec, num_time = sig.shape
    noise_sig = np.zeros((num_elec, num_time))
    # if 'is_highpassed' in args_params and args_params['is_highpassed']:
    #     sig_power = np.square(np.linalg.norm(highpass(sig,80,500), axis=1))/num_time
    # else:
    sig_power = np.square(np.linalg.norm(sig, axis=1))/num_time
    # if args_params is None:
        # Only add Gaussian noise
    for i in range(num_elec):
        noise_power = 10 ** (-(snr / 10)) * sig_power[i] / 2
        noise_std = np.sqrt(noise_power)
        noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,))
    # else:
    #     # Add realistic and Gaussian noise
    #     rnpower = args_params['rnpower']/num_time
    #     rndata = args_params['rndata']
    #     select_id = np.random.randint(0, rndata.shape[0])
    #     for i in range(num_elec):
    #         noise_power = 10 ** (-(snr / 10)) * sig_power[i]
    #         rpower = args_params['ratio']*noise_power                                 # realistic noise power
    #         noise_std = np.sqrt(noise_power - rpower)
    #         noise_sig[i, :] = sig[i, :] + np.random.normal(0, noise_std, (num_time,)) + np.sqrt(rpower/rnpower[select_id][i])*rndata[select_id][:, i]
    return noise_sig


def fwdJ_to_cortexJ(recon, rm):
    """
    :param recon: np.array; DeepSIF output, (num_time, num_region)
    :param rm: np.array; region mapping for each index, (num_vertices, )
    :return: J: np.array; DeepSIF output for each vertices, (num_time, num_vertices)
    """
    num_time, num_region = recon.shape
    num_vertices = rm.shape[0]
    J = np.zeros((num_time, num_vertices))
    for k in range(num_time):
        for i in range(num_region):
            J[k, rm==i] = recon[k, i]
    return J

def highpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5, axis: int=-1):
    sos = signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data,axis=axis)
    return filtered_data

def lowpass(data: np.ndarray, cutoff: float, sample_rate: float, poles: int = 5, axis: int=-1):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data,axis=axis)
    return filtered_data

def bandpass(data: np.ndarray, low_cutoff: float, high_cutoff: float, sample_rate: float, poles: int = 5, axis: int=-1):
    sos = signal.butter(poles, [low_cutoff, high_cutoff], 'bandpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data, axis=axis)
    return filtered_data

def compute_envelope(filtered_data):
    analytic_signal = signal.hilbert(filtered_data)
    envelope = np.abs(analytic_signal)
    return envelope

def detect_hfo(data, freq=80, threshold_factor=2):
    # Step 1: Bandpass filter the data in HFO frequency range
    filtered_data = highpass(data, freq , 500,axis=0)
    
    # Step 2: Compute the envelope of the filtered data
    envelope = compute_envelope(filtered_data)
    
    # Step 3: Set a threshold for HFO detection
    threshold = np.mean(envelope) + threshold_factor * np.std(envelope)
    
    # Step 4: Detect segments where the envelope exceeds the threshold
    hfo_mask = envelope > threshold
    hfo_segments = []
    
    # Identify continuous segments where HFO is detected
    start_idx = None
    for i in range(len(hfo_mask)):
        if hfo_mask[i] and start_idx is None:
            start_idx = i
        elif not hfo_mask[i] and start_idx is not None:
            end_idx = i
            hfo_segments.append((start_idx, end_idx))
            start_idx = None
    
    # Add the last segment if it ended with an HFO
    if start_idx is not None:
        hfo_segments.append((start_idx, len(hfo_mask)))
    
    return hfo_segments, envelope, threshold

def add_pink_noise_to_channels(signal, snr_target, alpha=1):
    """
    Add pink noise to each channel of a multi-channel signal independently.

    Args:
        signal: Input signal (channels x samples).
        snr_target: Target signal-to-noise ratio (in dB).
        alpha: Exponent for 1/f^alpha noise (default is 1 for pink noise).

    Returns:
        noisy_signal: Signal with pink noise added independently to each channel.
    """
    n_channels, n_samples = signal.shape
    noisy_signal = np.zeros_like(signal)  # Initialize noisy signal array

    for ch in range(n_channels):
        # Generate pink noise and scale it to match the desired SNR
        channel_signal = signal[ch, :]
        noisy_channel, _ = add_noise_with_snr(channel_signal, snr_target, noise_type='pink', alpha=alpha)
        noisy_signal[ch, :] = noisy_channel

    return noisy_signal
    
def add_noise_with_snr(signal, snr_target, noise_type='white', alpha=1):
    """
    Add noise to a signal to achieve a target SNR.
    
    Args:
        signal: The input signal array.
        snr_target: Desired signal-to-noise ratio (in dB).
        noise_type: Type of noise ('white' or 'pink').
        alpha: Exponent for 1/f^alpha noise (used for pink noise, default 1).
    
    Returns:
        Noisy signal and the noise added.
    """
    # Signal power
    power_signal = np.mean(signal**2)
    
    # Desired noise power
    power_noise = power_signal / (10**(snr_target / 10))
  
    # Generate noise
    if noise_type == 'white':
        noise = np.random.normal(0, 1, len(signal))  # White noise
    elif noise_type == 'pink':
        noise = generate_pink_noise(len(signal), alpha)  # Pink noise
    else:
        raise ValueError("Unsupported noise type. Use 'white' or 'pink'.")
    
    # Scale noise to match desired power
    current_noise_power = np.mean(noise**2)
    noise = noise * np.sqrt(power_noise / current_noise_power)
    
    # Add noise to the signal
    noisy_signal = signal + noise
    
    return noisy_signal, noise
    
def generate_pink_noise(n_samples, alpha=1):
    """
    Generate pink noise using the power-law distribution.
    Args:
        n_samples: Number of time samples.
        alpha: Exponent for 1/f^alpha noise (1 for pink noise).
    Returns:
        Pink noise signal of length n_samples.
    """
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero
    # Generate random phases and scale amplitudes by 1/f^alpha
    amplitudes = 1 / freqs**(alpha / 2)
    phases = np.random.uniform(0, 2 * np.pi, len(freqs))
    spectrum = amplitudes * np.exp(1j * phases)
    # Transform back to time domain
    signal = np.fft.irfft(spectrum)
    return signal / np.std(signal)  # Normalize to unit standard deviation
