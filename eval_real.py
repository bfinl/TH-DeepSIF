import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import collections

import torch


import network as Network

from utils import get_otsu_regions,highpass,lowpass,fwdJ_to_cortexJ,bandpass_fir_mne
from scipy.signal import hilbert
from scipy.spatial.distance import cdist


def pac_phasor_weights(
    x, fs,
    mid_point,
    hfo_halfwin_samp=50,      # 100 samples = 200 ms at 500 Hz
    low_band=(4.0, 30.0),     # low-freq phase carrier (paper convention); 4–30 Hz uses IIR on 0.5s segments
    ripple_band=(80.0, 200.0),# choose 80–160 (or 80–200) for 500 Hz fs
    eps=1e-12,
    combine_with_amp=True,
    amp_stat="max",           # "max" or "mean" amplitude inside window
):
    """
    Compute per-channel PAC weights using an amplitude-weighted phasor:
        v = | Σ_t a(t) * exp(1j * phi(t)) | / Σ_t a(t)
    where phi(t) is low-band phase and a(t) is ripple-band amplitude
    over the HFO window [mid_point - hfo_halfwin_samp, mid_point + hfo_halfwin_samp). Adjust via hfo_halfwin_samp or --pac_hfo_halfwin.

    Low band: PAC literature (e.g. epilepsy iEEG) typically uses phase of
    low-frequency activity <30 Hz (e.g. 4-30 Hz). For f_lo < 8 Hz we use IIR
    so 4-30 Hz works on 0.5 s segments.

    Returns
    -------
    w : (C,) in [0,1]
        Coupling-based weights (optionally multiplied by ripple amplitude).
    details : dict
        Diagnostics: vector strength v and preferred phase theta.
    """

    # Ensure 2D: (T, C)
    if x.ndim == 1:
        x = x[:, None]
    T, C = x.shape

    # Window indices (clip to valid range)
    s0 = max(0, mid_point - hfo_halfwin_samp)
    s1 = min(T, mid_point + hfo_halfwin_samp)

    # bandpass_fir_mne expects (C, T); x is (T, C)
    # 1) Low-frequency component -> phase phi(t)
    x_low = bandpass_fir_mne(x.T, fs, low_band[0], low_band[1]).T  # (T, C)
    phi = np.angle(hilbert(x_low, axis=0))  # (T, C)

    # 2) Ripple-band component -> amplitude a(t)
    x_rip = bandpass_fir_mne(x.T, fs, ripple_band[0], ripple_band[1]).T  # (T, C)
    a = np.abs(hilbert(x_rip, axis=0))      # (T, C)

    # Restrict to HFO window
    phi_w = phi[s0:s1, :]   # (W, C)
    a_w   = a[s0:s1, :]     # (W, C)

    # Amplitude-weighted phasor for each channel
    z = np.sum(a_w * np.exp(1j * phi_w), axis=0)  # (C,)
    denom = np.sum(a_w, axis=0) + eps
    v = np.abs(z) / denom                          # vector strength in [0,1]
    theta = np.angle(z)                            # preferred phase (radians)

    # Optional: incorporate ripple amplitude (paper notes coupling correlates with ripple amplitude)  (https://pmc.ncbi.nlm.nih.gov/articles/PMC5118142/)
    if combine_with_amp:
        if amp_stat == "max":
            amp = np.max(a_w, axis=0)
        elif amp_stat == "mean":
            amp = np.mean(a_w, axis=0)
        else:
            raise ValueError("amp_stat must be 'max' or 'mean'")
        amp = amp / (np.max(amp) + eps)
        w = v * amp
    else:
        w = v.copy()

    # Normalize to [0,1] for multiplicative gating
    w = w / (np.max(w) + eps)

    details = {"v": v, "theta": theta, "win": (s0, s1)}
    return w, details

def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--use_pc', type=int, default=True, metavar='t/f', help='if we are using pc for training')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--dat', default='SpikeEEGBuildEval', type=str, help='data loader')
    parser.add_argument('--patient_id', type=str, help='patient id')
    parser.add_argument('--model_id', type=int, default=49884, help='model id')
    parser.add_argument('--num_electrode', type=int, default=49884, help='number of electrode')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    parser.add_argument('--arch', default='TemporalInverseNet', type=str, help='network')
    parser.add_argument('--num_layer', default=6, type=int, help='number of rnn layer')

    parser.add_argument('--snr_rsn_ratio', default=0, type=float, help='ratio between real noise and gaussian noise')
    parser.add_argument('--lfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')
    parser.add_argument('--hfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')

    parser.add_argument('--inv_lfd', default=False, type=int, help='forward model to use')
    parser.add_argument('--scale_mag', default=False, type=int, help='forward model to use')
    parser.add_argument('--spike_type', default='p', type=str, help='select n,p,r spike for analysis')
    parser.add_argument('--patient_dataset', default='', type=str, help='pateint_xxx, select xxx here')
    parser.add_argument('--patient_root', default='', type=str,
                        help="directory holding patient<id>/data_75.mat; "
                             "defaults to patient_individual_spikes/patient_<patient_dataset>. "
                             "Use --patient_root example_data to run the bundled example.")
    parser.add_argument('--highpass_eeg', dest='highpass_eeg', default=0, type=int, help='highpass cutoff (Hz) for EEG/HFO analysis; 0 = no highpass')
    parser.add_argument('--suppress_distant_hfo', default=0, type=int, help='suppress high frequency in channels far from spike channels (other hemisphere). If > 0, value is the HFO cutoff frequency (default 80 Hz)')
    parser.add_argument('--output_folder', default='', type=str, help='name of folder inside model directory to save output files (if empty, saves directly in model directory)')
    parser.add_argument('--remove_pHFO', default=False, type=int, help='zero out pHFO channels from input using pHFO_channels.mat')
    parser.add_argument('--pHFO_file', default='example_data/pHFO_channels.mat', type=str, help='path to pHFO_channels.mat file (1x140 cell)')
    parser.add_argument('--n_neighbors', default=0, type=int, help='number of nearest neighbor channels to also zero out around each pHFO channel')
    parser.add_argument('--eloc_file', default='anatomy/75_channels_electrode.mat', type=str, help='electrode position file for neighbor finding')
    parser.add_argument('--remove_hf_noise', default=False, type=int, help='remove channels with higher HF energy than pHFO channels (requires --remove_pHFO and highpass_eeg)')
    parser.add_argument('--hf_threshold_mode', default='phfo_max', type=str, choices=['phfo_max', 'percentile', 'median'], help='threshold mode: phfo_max = max pHFO energy, percentile = Nth percentile, median = median of all channels')
    parser.add_argument('--hf_percentile', default=75, type=float, help='percentile threshold when hf_threshold_mode=percentile')

    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = (True) and torch.cuda.is_available()  # Only use GPU during training
    device = torch.device(args.device if use_cuda else "cpu")
    
    if args.use_pc:
        data_root = 'source/'
        result_root = 'model_weights/{}_the_model'.format(args.model_id)
        dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']

    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return
    
    # Create output folder if specified
    if args.output_folder:
        output_dir = os.path.join(result_root, args.output_folder)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = result_root
  
    # ================================== LOAD pHFO channels if requested =============================================================================
    if args.remove_pHFO or args.remove_hf_noise:
        pHFO_data = loadmat(args.pHFO_file, squeeze_me=False)
        # pHFO_channels is a 1x140 cell array; each cell contains channel indices (MATLAB 1-based or 0-based as stored)
        pHFO_cells = pHFO_data['pHFO_channels']  # shape (1, 140)
        pHFO_cells = pHFO_cells.flatten()  # shape (140,)
        print(f"Loaded pHFO_channels with {len(pHFO_cells)} events")

        # Load electrode positions for neighbor finding if requested
        if args.n_neighbors > 0 or args.remove_hf_noise:
            eloc_data = loadmat(args.eloc_file)
            channels = eloc_data['eloc'][0, 0]['Channel'][0]
            eloc_pos_3d = np.array([ch[2].flatten() for ch in channels])  # (75, 3)
            eloc_dist_matrix = cdist(eloc_pos_3d, eloc_pos_3d)
            print(f"Loaded electrode positions for neighbor finding (n_neighbors={args.n_neighbors})")

    global_event_idx = 0  # tracks event index across all patients

    # ================================== LOAD DATA ===================================================================================================

    patient_ids = args.patient_id.split(",")
    for p in patient_ids:

        patient_root = args.patient_root or os.path.join(
            "patient_individual_spikes", "patient_" + args.patient_dataset)
        patientdata = loadmat(os.path.join(patient_root, "patient" + str(p), "data_75.mat"))
        data = patientdata["sigEpochSpike_"+args.spike_type];

        data_all = []
        for cluster in range(int(data.shape[0])):
            
            print(data[cluster,0].shape)
            if len(data[cluster,0].shape)<3:
                data_all.append(np.expand_dims(data[cluster,0],axis=2))
            else:
                data_all.append(data[cluster,0])
        
        data = np.concatenate(data_all,axis = 2)
        # Cap at 15 events per patient to match MATLAB pHFO_channels.mat generation
        if data.shape[2] > 15:
            print(f"  Patient {p}: capping events from {data.shape[2]} to 15")
            data = data[:, :, :15]

        print(data.shape)

        EEG = np.zeros([500,data.shape[0],data.shape[2]])
        reconstruction = np.zeros([500,994,data.shape[2]])
        for i in range(int(data.shape[2])):


            # data is (n_channels, n_times, n_spikes); keep (n_channels, n_times) for MNE (time last)
            individual_spike = np.squeeze(data[:, :500, i]).astype(np.float64)
            individual_spike = individual_spike / (np.max(np.abs(individual_spike)))

            n_times = individual_spike.shape[1]
            mid_point = n_times // 2
            spike_window = individual_spike[:, mid_point-128:mid_point+128]

            individual_hfo = bandpass_fir_mne(individual_spike, 500, args.highpass_eeg, 200)
            hfo_window = individual_hfo[:, mid_point-50:mid_point+50]
          
            # Original weight definition (spike-window amplitude only)
            # hfo_window, spike_window are (n_channels, n_time); max over time -> (n_channels,)
            hfo_weights = np.max(np.abs(hfo_window), axis=1) / (np.max(np.abs(hfo_window)) + 1e-12)
            spike_weights = np.max(np.abs(spike_window), axis=1)
            weights = spike_weights
            weights = weights / np.max(weights)


            

            print(args.highpass_eeg)
            if args.highpass_eeg > 0 :
                individual_spike = bandpass_fir_mne(individual_spike, 500, args.highpass_eeg, 200)

            # Remove HFO from pHFO channels by zeroing them after highpass filtering
            if args.remove_pHFO:
                if global_event_idx < len(pHFO_cells):
                    phfo_ch = pHFO_cells[global_event_idx]
                    if phfo_ch is not None and phfo_ch.size > 0:
                        phfo_idx = phfo_ch.flatten().astype(int)
                        phfo_idx = phfo_idx - 1  # MATLAB 1-based -> 0-based
                        # Find neighbor channels if requested
                        if args.n_neighbors > 0:
                            neighbor_set = set()
                            center_set = set(phfo_idx.tolist())
                            for ci in phfo_idx:
                                sorted_idx = np.argsort(eloc_dist_matrix[ci])
                                count = 0
                                for ni in sorted_idx:
                                    if ni == ci or ni in center_set:
                                        continue
                                    neighbor_set.add(ni)
                                    count += 1
                                    if count >= args.n_neighbors:
                                        break
                            all_zero_idx = np.unique(np.concatenate([phfo_idx, np.array(list(neighbor_set), dtype=int)]))
                            print(f"  Event {global_event_idx}: zeroing {len(phfo_idx)} pHFO + {len(neighbor_set)} neighbor channels: {all_zero_idx}")
                            individual_spike[all_zero_idx, :] = 0.0
                        else:
                            print(f"  Event {global_event_idx}: zeroing {len(phfo_idx)} pHFO channels: {phfo_idx}")
                            individual_spike[phfo_idx, :] = 0.0
                    else:
                        print(f"  Event {global_event_idx}: no pHFO channels to remove")
                else:
                    print(f"  WARNING: global_event_idx {global_event_idx} exceeds pHFO_cells length {len(pHFO_cells)}")
                global_event_idx += 1

            # Remove channels with higher HF energy than pHFO channels
            if args.remove_hf_noise:
                cur_idx = global_event_idx if not args.remove_pHFO else global_event_idx - 1
                if cur_idx < len(pHFO_cells):
                    phfo_ch = pHFO_cells[cur_idx]
                    if phfo_ch is not None and phfo_ch.size > 0:
                        phfo_idx_noise = phfo_ch.flatten().astype(int) - 1
                        # Compute HF energy per channel on the current (already bandpassed 80-200) signal
                        n_t = individual_spike.shape[1]
                        mid_t = n_t // 2
                        half_win = int(0.05 * 500)  # 25 samples = 50ms, matches topo visualization
                        hf_window = individual_spike[:, mid_t - half_win:mid_t + half_win]
                        hf_energy = np.sum(hf_window ** 2, axis=1)
                        if args.hf_threshold_mode == 'percentile':
                            threshold = np.percentile(hf_energy, args.hf_percentile)
                        elif args.hf_threshold_mode == 'median':
                            threshold = np.median(hf_energy)
                        else:
                            threshold = np.max(hf_energy[phfo_idx_noise])
                        # Find nearest neighbors of pHFO channels to protect
                        # If multiple pHFO channels, only protect neighbors shared by ALL pHFO channels
                        n_protect = args.n_neighbors if args.n_neighbors > 0 else 4
                        protected_set = set(phfo_idx_noise.tolist())
                        phfo_list = phfo_idx_noise.tolist()
                        if len(phfo_list) == 1:
                            # Single pHFO channel: protect its N closest neighbors
                            sorted_idx = np.argsort(eloc_dist_matrix[phfo_list[0]])
                            count = 0
                            for ni in sorted_idx:
                                if ni in protected_set:
                                    continue
                                protected_set.add(ni)
                                count += 1
                                if count >= n_protect:
                                    break
                        else:
                            # Multiple pHFO channels: for each, get N closest neighbors, then intersect
                            neighbor_sets = []
                            for ci in phfo_list:
                                sorted_idx = np.argsort(eloc_dist_matrix[ci])
                                neighbors = []
                                for ni in sorted_idx:
                                    if ni in set(phfo_list):
                                        continue
                                    neighbors.append(ni)
                                    if len(neighbors) >= n_protect:
                                        break
                                neighbor_sets.append(set(neighbors))
                            shared = neighbor_sets[0]
                            for ns in neighbor_sets[1:]:
                                shared = shared & ns
                            protected_set.update(shared)
                            print(f"  Event {cur_idx}: {len(phfo_list)} pHFO channels, {len(shared)} shared neighbors protected")
                        noise_channels = np.where(hf_energy > threshold)[0]
                        noise_channels = np.array([ch for ch in noise_channels if ch not in protected_set])
                        if len(noise_channels) > 0:
                            individual_spike[noise_channels, :] = 0.0
                            print(f"  Event {cur_idx}: zeroing {len(noise_channels)} HF noise channels (threshold={threshold:.6f}, mode={args.hf_threshold_mode})")
                        else:
                            print(f"  Event {cur_idx}: no HF noise channels above threshold ({args.hf_threshold_mode}={threshold:.6f})")
                if not args.remove_pHFO:
                    global_event_idx += 1

            # Suppress HFO in distant channels using PAC phasor weights (epi.13572-style)
            if args.suppress_distant_hfo > 0:
                individual_spike = individual_spike * weights[:, np.newaxis]  # (75,) -> (75,1) to broadcast with (75, n_times)
            
            
            # Model expects (batch, time, channels) = (1, n_times, 75); individual_spike is (75, n_times)
            input_spike = torch.tensor(np.expand_dims(individual_spike.T, axis=0)).float()
            input_spike = input_spike.to(device)

            
            # =============================== LOAD MODEL =====================================================================================================

            if args.resume:
                fn = fn = os.path.join(result_root, 'epoch_' + args.resume)
            else:
                fn = os.path.join(result_root, 'model_best.pth.tar')
                
            print("=> Load checkpoint", fn)
            if os.path.isfile(fn):
                print("=> Found checkpoint '{}'".format(fn))
                checkpoint = torch.load(fn, map_location=torch.device('cpu'))
                # best_result = checkpoint['best_result']
                # net = network.__dict__[checkpoint['arch']](*checkpoint['attribute_list']).to(device)  # redefine the weights architecture
                if args.arch == 'TemporalInverseNet':
                    # Legacy DeepSIF RNN checkpoint. NetworkRNN.py is not shipped
                    # with this repository; obtain it from https://github.com/bfinl/DeepSIF
                    import NetworkRNN as network
                    net = network.__dict__['TemporalInverseNet'](*checkpoint['attribute_list']).to(device)
                else:
                    
                    net = Network.__dict__[args.arch](input_channels = args.num_electrode,num_layers=args.num_layer).to(device)  # redefine the weights architecture
                net.load_state_dict(checkpoint['state_dict'], strict=False)

                # Define logger
                logger = logging.getLogger(__name__)
                logger.setLevel(logging.INFO)
                handler = logging.FileHandler(result_root + '/outputs_eval_sim.log')
                handler.setLevel(logging.INFO)
                logger.addHandler(handler)
                logger.info("=================== Evaluation mode: {} ====================================".format(datetime.datetime.now()))
                logger.info("Patient is {}".format(args.patient_id))
                # Save every parameters in args
                for v in args.__dict__:
                    if v not in ['workers', 'train', 'test']:
                        logger.info('{} is {}'.format(v, args.__dict__[v]))
            else:
                print("ERROR: no checkpoint found")
                return

            # =============================== EVALUATION =====================================================================================================
            net.eval()

            eval_dict = collections.defaultdict(list)
            with torch.no_grad():

                if args.arch == 'TemporalInverseNet':
                    model_output = net(input_spike)
                    out = model_output['last']
                else:
                    model_output = net(input_spike.transpose(0,1))
                    out = model_output

            
            # Save to output folder with base filename from checkpoint
            base_name = os.path.basename(fn).replace('.pth.tar', '').replace('epoch_', '')
            output_filename = os.path.join(output_dir, '{}_preds_patient{}{}_{}_spike{}.mat'.format(base_name, p, args.info, args.spike_type, i+1))
            savemat(output_filename, {"EEG":input_spike.cpu().numpy(),"reconstruction":out.cpu().numpy()})

        #     reconstruction[:,:,i] = out.cpu().numpy()
        #     EEG[:,:,i] = input_spike.cpu().numpy()
        

        # np.save(fn + '_preds_patient{}{}_{}_individualspikes.npy'.format(p, args.info,args.spike_type), reconstruction)
    
    # Create eval_params.txt file for manual MATLAB execution
    if args.patient_id and args.output_folder:
        patient_ids_list = [p.strip() for p in args.patient_id.split(",")]
        
        # Generate unique filename to avoid conflicts when running multiple processes
        # Use SLURM_JOB_ID if available, otherwise use process ID and timestamp
        unique_id = os.environ.get('SLURM_JOB_ID', '')
        if not unique_id:
            unique_id = f"{os.getpid()}_{int(time.time())}"
        
        # Write patient IDs and parameters to a file for MATLAB script
        eval_params_file = os.path.join(output_dir, f'eval_params_{unique_id}.txt')
        with open(eval_params_file, 'w') as f:
            f.write(f"MODEL={args.model_id}\n")
            f.write(f"PATIENT_IDS={','.join(patient_ids_list)}\n")
            f.write(f"SPIKE_TYPE={args.spike_type}\n")
            f.write(f"DATASET={args.info if args.info else ''}\n")
            f.write(f"OUTPUT_DIR={output_dir}\n")
            f.write(f"RESULT_ROOT={result_root}\n")
            f.write(f"PATIENT_DATASET={args.patient_dataset}\n")
        
        # Also save the parameter file path to a marker file for easy retrieval
        import shutil
        project_root = os.getcwd()
        marker_file = os.path.join(project_root, f'eval_params_{unique_id}.txt')
        shutil.copy2(eval_params_file, marker_file)
        
        print(f"\n=== Evaluation parameters saved ===")
        print(f"Parameters file created at: {eval_params_file}")
        print(f"Also copied to project root: {marker_file}")
        print(f"\nTo run MATLAB evaluation manually, execute:")
        print(f"  cd {project_root}")
        print(f"  matlab -nodisplay -nosplash -r \"run('patient_individual_spikes/debug.m'); exit;\"")
        print(f"\nOr specify the parameter file:")
        print(f"  matlab -nodisplay -nosplash -r \"eval_params_file='{marker_file}'; run('patient_individual_spikes/debug.m'); exit;\"")


if __name__ == '__main__':
    main()

