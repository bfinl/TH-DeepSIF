import os
import sys
import numpy as np
import multiprocessing as mp
import mne
from mne.datasets import sample
from mne.inverse_sparse import mixed_norm, make_stc_from_dipoles
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, apply_inverse
from mne.beamformer import make_lcmv, apply_lcmv
from scipy.io import loadmat, savemat
from mne import create_info
import glob
from skimage.filters import threshold_otsu
import time
import argparse
from scipy import signal
import sys
import torch
import collections
import hdf5storage
import random
# Add the parent directory to the path to import loaders and utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from loaders import HFOEEGBuildEvalJR
from utils import get_otsu_regions


# 2021 May: Run mne ESI code
# 2023 Feb: Clean up the code
# 2024: Modified to use HFOEEGBuildEvalJR loader and evaluate with get_otsu_regions

def log_message(message, log_file='mne_progress_E4s.log'):
    """Write message to both console and log file with timestamp"""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    
    # Write to log file
    with open(log_file, 'a') as f:
        f.write(full_message + '\n')
        f.flush()  # Force write to disk immediately

def cal_mne(forward, srcmat, args):
    """
    Modified cal_mne function that uses HFOEEGBuildEvalJR loader instead of reading patient data directly
    """
    
    log_message("Starting cal_mne function...")
    
    # Load distance matrix for evaluation
    try:
        dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']
        log_message("Loaded distance matrix successfully")
    except:
        log_message("Warning: Could not load distance matrix. Evaluation metrics will not include location error.")
        dis_matrix = None
    
    # Initialize evaluation dictionary
    eval_dict = collections.defaultdict(list)
    eval_dict['all_regions'] = []
    eval_dict['all_out'] = []
    eval_dict['all_nmm'] = []
    
    # Create dataset using HFOEEGBuildEvalJR
    args_params = {
        'use_pc': args.use_pc,
        'prob_add_spike': args.prob_add_spike if hasattr(args, 'prob_add_spike') else 0,
        'highpassed_eeg': args.highpassed_eeg if hasattr(args, 'highpassed_eeg') else 0,
        'is_normalized': args.is_normalized if hasattr(args, 'is_normalized') else 0,
        'variable_length': args.variable_length if hasattr(args, 'variable_length') else 0
    }
    
    log_message("Loading forward matrix and other anatomy files...")
    
    fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd_small']
    dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']
    rm = loadmat('anatomy/fs_cortex_20k_region_mapping.mat')['rm']
    
    log_message("Anatomy files loaded successfully")
    
    inv_ind = [439, 441, 495, 492, 498, 420, 419, 409, 408, 415, 416, 417, 418, 419, 915, 916, 918, 917, 923, 910, 922, 919]
    fwd[:, inv_ind] = -fwd[:, inv_ind]

    log_message("Creating dataset with HFOEEGBuildEvalJR...")
    
    # Create dataset
    dataset = HFOEEGBuildEvalJR(args.data_root, fwd=fwd, transform=None, args_params=args_params)
    
    log_message(f"Dataset created successfully. Size: {len(dataset)}")
    
    log_message("Starting to process samples...")
    
    # Process each sample
    for idx in range(1000):  # Only process first 100 samples
        if idx % 10 == 0:
            log_message(f"Processing sample {idx}/{len(dataset)}")
            
        # Add debug info for first sample
        if idx == 0:
            log_message(f"Processing first sample...")
            
        # Get sample data
        sample = dataset[idx*17]
        eeg_data = sample['data']  # Shape: (time, channels)
        nmm = sample['nmm']        # Shape: (time, regions)
        label = sample['label']    # Shape: (num_sources, max_size)
        
        if idx == 0:
            log_message(f"First sample data shapes - EEG: {eeg_data.shape}, NMM: {nmm.shape}, Label: {label.shape}")
        
        # Clear memory after each sample to prevent accumulation
        if idx > 0 and idx % 50 == 0:
            import gc
            gc.collect()
            log_message(f"Memory cleared at sample {idx}")
        
        # Convert to MNE format
        epoch_data = mne.EpochsArray(np.expand_dims(eeg_data.transpose(), 0), info, tmin=0)
        epoch_data = epoch_data.set_eeg_reference(ref_channels='average', projection=True)
        epoch_data.apply_proj()
        
        # Compute covariance matrices
        data_cov = mne.compute_covariance(epoch_data, tmin=0/1000, tmax=1000/1000, method='shrunk')
        noise_cov = mne.compute_covariance(epoch_data, tmin=0/1000, tmax=1000/1000, method='shrunk')
        noise_cov['data'] = np.diag(np.ones(eeg_data.shape[1],))  # No whitening
        
        evoked = epoch_data.average()
        
        # Apply sLORETA inverse solution
        alpha = 55  # regularization parameter between 0 and 100 (100 is high)
        loose = 0.9 # loose orientation & depth weighting
        depth = 0.9
        n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solver
        
        inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov, depth=depth, fixed=True, use_cps=True)
        stc_sloreta = apply_inverse(evoked, inverse_operator, lambda2=1. / 9., method='sLORETA')
        sloreta_J = stc_sloreta.data
        # Convert from 20484 vertices to 994 regions using existing mapping
        # rm is 1x20484 array with values 0-993
    
        sloreta_regions = np.zeros((sloreta_J.shape[1], 994))  # (time, 994 regions)
        
        for region_idx in range(994):
            # Find vertices belonging to this region
            vertex_indices = np.where(rm.flatten() == region_idx)[0]
            if len(vertex_indices) > 0:
                # Select the first vertex in this region
                selected_vertex = vertex_indices[0]
                # Extract activity for this vertex across time
                sloreta_regions[:, region_idx] = sloreta_J[ selected_vertex,:]
        
        # Reshape output to match expected format: (batch_size, time, regions)
        out = np.expand_dims(sloreta_regions, 0)  # Add batch dimension
        
        # Convert nmm and label to expected format
        nmm_batch = np.expand_dims(nmm, 0)  # Add batch dimension
        label_batch = np.expand_dims(label, 0)  # Add batch dimension
        
        # Evaluate using get_otsu_regions
        args_params_eval = {'dis_matrix': dis_matrix, 'hfo': True}  # Set hfo=True for HFO data
        eval_results = get_otsu_regions(out, nmm_batch, label_batch, args_params_eval)
        
        # Store results
        eval_dict['precision'].extend(eval_results['precision'])
        eval_dict['recall'].extend(eval_results['recall'])
        eval_dict['le'].extend(eval_results['le'])
        
        # Store all_regions and all_out (keep everything from get_otsu_regions)
        for kk in range(out.shape[0]):
            # Store detected regions - ensure it's a numpy array
            detected_regions = np.asarray(eval_results['all_regions'][kk])
            
            # Store sLORETA output for detected regions (all regions, no downsampling)
            if eval_results['all_out'][kk].size > 0:
                out_data = eval_results['all_out'][kk]  # Shape: (time, num_detected_regions)
                
                # Ensure detected_regions is integer type
                detected_regions = detected_regions.astype(int)
                
                eval_dict['all_out'].append(out_data)
                eval_dict['all_regions'].append(detected_regions)
            else:
                eval_dict['all_out'].append(np.array([]))
                eval_dict['all_regions'].append(np.array([], dtype=int))
            
            # Store NMM waveforms for ground truth sources only (1 or 2 sources)
            # Extract center region for each source from label
            num_sources = label_batch.shape[1]  # Number of sources (1 or 2)
            source_waveforms = []  # List to store waveforms for each source
            
            for src_idx in range(num_sources):
                # Extract GT regions for this source (0-indexed)
                source_regions = label_batch[kk, src_idx, :]
                # Filter out padding (15213) and invalid regions
                source_regions = source_regions[source_regions != 15213]
                source_regions = source_regions[source_regions >= 0]
                source_regions = source_regions[source_regions < nmm_batch.shape[2]]
                source_regions = np.unique(source_regions)
                
                if len(source_regions) > 0:
                    # Use center region (first region) for this source
                    center_region = int(source_regions[0])
                    # Extract NMM waveform at center region: (time,)
                    source_waveform = nmm_batch[kk, :, center_region]
                    source_waveforms.append(source_waveform)
            
            # Convert to array: (time, num_sources) or (time,) if single source
            if len(source_waveforms) > 0:
                if len(source_waveforms) == 1:
                    # Single source: store as (time,)
                    eval_dict['all_nmm'].append(source_waveforms[0])
                else:
                    # Multiple sources: store as (time, num_sources)
                    eval_dict['all_nmm'].append(np.column_stack(source_waveforms))
            else:
                # No valid sources found
                eval_dict['all_nmm'].append(np.array([]))
    
    log_message(f"Processing completed. Processed {len(eval_dict['precision'])} samples successfully.")
    
    # Calculate summary statistics
    precision_values = [p for p in eval_dict['precision'] if p is not None]
    recall_values = [r for r in eval_dict['recall'] if r is not None]
    le_values = [le for le in eval_dict['le'] if le is not None]
    
    if precision_values:
        eval_dict['mean_precision'] = np.mean(precision_values)
        eval_dict['std_precision'] = np.std(precision_values)
    if recall_values:
        eval_dict['mean_recall'] = np.mean(recall_values)
        eval_dict['std_recall'] = np.std(recall_values)
    if le_values:
        eval_dict['mean_le'] = np.mean(le_values)
        eval_dict['std_le'] = np.std(le_values)
    
    
    # Print summary
    log_message(f"\nEvaluation Summary:")
    if 'mean_precision' in eval_dict:
        log_message(f"Precision: {eval_dict['mean_precision']:.3f} ± {eval_dict['std_precision']:.3f}")
    if 'mean_recall' in eval_dict:
        log_message(f"Recall: {eval_dict['mean_recall']:.3f} ± {eval_dict['std_recall']:.3f}")
    if 'mean_le' in eval_dict:
        log_message(f"Location Error: {eval_dict['mean_le']:.3f} ± {eval_dict['std_le']:.3f}")
  
    # Save results
    output_filename = f'patient_hfo/TraditionalMethod/s_sloreta_eval_results_{args.data_root.split("/")[-1].replace(".mat", "")}.mat'
    
    # Convert lists to object arrays for MATLAB cell array compatibility
    for key in ['all_regions', 'all_out', 'all_nmm']:
        if key in eval_dict and isinstance(eval_dict[key], list) and len(eval_dict[key]) > 0:
            # Create object array explicitly to handle arrays of different shapes
            obj_array = np.empty(len(eval_dict[key]), dtype=object)
            for i, item in enumerate(eval_dict[key]):
                obj_array[i] = item
            eval_dict[key] = obj_array

    hdf5storage.savemat(output_filename, eval_dict, format=7.3, matlab_compatible=True, compress=False )
    log_message(f"Results saved to {output_filename}")
        
    return eval_dict

def myisnan(x):
    # identify the padding in array
    return np.abs(x - 15213) < 1e-6

if __name__ == '__main__':

    mne.set_log_level('CRITICAL')
    os.environ["MKL_NUM_THREADS"] = "1"

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MNE sLORETA ESI Evaluation with HFOEEGBuildEvalJR')
    parser.add_argument('--use_pc', type=int, default=1, help='Path to dataset metadata file (.mat)')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset metadata file (.mat)')
    parser.add_argument('--prob_add_spike', type=float, default=0, help='Probability of adding spike')
    parser.add_argument('--highpassed_eeg', type=int, default=0, help='High-pass filter cutoff frequency')
    parser.add_argument('--is_normalized', type=int, default=0, help='Normalization flag')
    parser.add_argument('--variable_length', type=int, default=0, help='Variable length flag')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    args = parser.parse_args()

    # ==============================  Create info file  ==================================================================
    ch_names = []
    for i in range(75):
        ch_names.append('MEG{}'.format(i))
    info = create_info(ch_names, 500, ['eeg'] * len(ch_names))

    # ==============================  Load my own forward ==================================================================
    data_path = sample.data_path()
    
    fwd_fname = str(data_path)+ "/" + '/MEG/sample/sample_audvis-eeg-oct-6-fwd.fif'
    forward = mne.read_forward_solution(fwd_fname)
    lfd = loadmat('anatomy/leadfield_75_20k.mat')['lfd_free']  # need to load the free orientation file
    nchannel = 75
    srcmat = loadmat('anatomy/fs_cortex_20k.mat')
    lfd = np.transpose(lfd)
    lfd = lfd.reshape((-1, 3, nchannel))
    lfd_new = np.copy(lfd)
    lfd_new[:srcmat['left_pos'].shape[0], :, :] = lfd[srcmat['left_idx'].reshape(-1)-1, :, :]
    lfd_new[srcmat['left_pos'].shape[0]:, :, :] = lfd[srcmat['right_idx'].reshape(-1)-1, :, :]
    lfd_new = np.transpose(lfd_new.reshape(-1,lfd.shape[2]))
    
    forward['nsource'] = 20484
    forward['nchan'] = nchannel
    forward['sol']['data'] = lfd_new
    forward['sol']['nrow'] = nchannel
    forward['sol']['ncol'] = 61452
    forward['sol']['row_names'] = ch_names
    forward['_orig_sol'] = lfd_new
    forward['sol_grad'] = None
    forward['mri_head_t'] = None
    forward['info'] = info
    
    # Prepare source space
    src_list = [{'rr': srcmat['left_pos'], 'tris':srcmat['left_tri']-1, 'nn':srcmat['left_ori'],
                'vertno':np.arange(srcmat['left_pos'].shape[0]),'type':'surf', 'nuse':srcmat['left_idx'].size},
                {'rr': srcmat['right_pos'], 'tris':srcmat['right_tri']-1, 'nn':srcmat['right_ori'],
                'vertno':np.arange(srcmat['right_pos'].shape[0]),'type':'surf','nuse':srcmat['right_idx'].size}]

    forward['src'] = mne.SourceSpaces(src_list)
    forward['source_rr'] = np.concatenate((srcmat['left_pos'],srcmat['right_pos']))
    forward['_orig_source_ori'] = None
    forward['source_nn'] = np.tile(np.array([[1,0,0],[0,1,0],[0,0,1]]),(forward['nsource'],1))
    forward['surf_ori'] = None

    # Run evaluation
    print("Starting MNE sLORETA evaluation...")
    eval_results = cal_mne(forward, srcmat, args)
    print('Evaluation completed successfully!') 