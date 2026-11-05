import os
import sys
import sys
# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
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
from utils import highpass


# 2021 May: Run mne ESI code
# 2023 Feb: Clean up the code

def cal_distributed(J, peak_time,time):
    print(J.shape)
    recon_time = np.zeros((len(peak_time)*2, time))
    J_recon = np.zeros((len(peak_time), J.shape[0]))
    for i in range(len(peak_time)):
        J_tmp = np.abs(J[:, peak_time[i]])
        J_tmp  = J_tmp - np.min(J_tmp)
        J_tmp = J_tmp/np.max(J_tmp)
        thresh = threshold_otsu(J_tmp, nbins=100)
        print(thresh)
        print(J_tmp.shape)
        recon_time[i] =  np.mean(J[J_tmp>thresh,:], axis=0)
        recon_time[i+len(peak_time)] =  np.mean(J[J_tmp>0.9,:], axis=0)
        J_recon[i] = J_tmp.transpose()
    return J_recon, recon_time

def cal_mne(forward, srcmat):
    global fname
    global all_time_used

    parser = argparse.ArgumentParser(description='Rachel DLSL Model')
    parser.add_argument('--patient_id', type=str, help='patient id')
    parser.add_argument('--reg', type=float, default=1, metavar='t/f', help='regulation for LCMV')
    parser.add_argument('--weight_norm', type=str, default='unit-noise-gain', metavar='t/f', help='Method for the Weighted Norm')
    parser.add_argument('--pick_ori', type=str, default='None', metavar='t/f', help='Method for pick orientation')
    parser.add_argument('--output_folder', type=str, default='patient_sloreta', help='Output folder')
    parser.add_argument('--suppress_distant_hfo', type=int, default=0, help='Suppress distant HFO')
    parser.add_argument('--remove_hf_noise', type=int, default=0, help='Remove HF noise channels')
    parser.add_argument('--hf_threshold_mode', type=str, default='phfo_max',
                        choices=['phfo_max', 'percentile', 'median'], help='HF noise threshold mode')
    parser.add_argument('--hf_percentile', type=float, default=75, help='Percentile for threshold')
    parser.add_argument('--n_neighbors', type=int, default=8, help='Number of protected neighbors')
    parser.add_argument('--pHFO_file', type=str,
                        default='patient_individual_spikes/patient_pHFO/pHFO_channels.mat')
    parser.add_argument('--eloc_file', type=str,
                        default='traditional_methods_matlab/75_channels_electrode.mat')

    args = parser.parse_args()

    # Load pHFO and electrode data if HF noise removal is enabled
    pHFO_cells = None
    eloc_dist_matrix = None
    if args.remove_hf_noise:
        from scipy.spatial.distance import cdist
        pHFO_data = loadmat(args.pHFO_file, squeeze_me=False)
        pHFO_cells = pHFO_data['pHFO_channels'].flatten()
        eloc_data = loadmat(args.eloc_file)
        channels = eloc_data['eloc'][0, 0]['Channel'][0]
        eloc_pos = np.array([ch[2].flatten() for ch in channels])
        eloc_dist_matrix = cdist(eloc_pos, eloc_pos)

    rm = loadmat('anatomy/fs_cortex_20k_region_mapping.mat')['rm']
    rm = rm.flatten()
    # Iterate through all data
    patient_ids = args.patient_id.split(",")
    global_event_idx = 0
    for p in patient_ids:
        # # ==============================  Create my own epoched data ===================================================
        patientdata = loadmat("patient_individual_spikes/patient_pHFO/patient"+str(p)+"/data_75.mat")
        data = patientdata["sigEpochSpike_p"]

        print("Patient ID: ", p)
        data_all = []
        for cluster in range(int(data.shape[0])):
            
            print(data[cluster,0].shape)
            if len(data[cluster,0].shape)<3:
                data_all.append(np.expand_dims(data[cluster,0],axis=2))
            else:
                data_all.append(data[cluster,0])
        
        data = np.concatenate(data_all,axis = 2)
        if data.shape[2] > 15:
            print(f"  Patient {p}: capping events from {data.shape[2]} to 15")
            data = data[:, :, :15]

        print(data.shape)

        num_hfo = data.shape[2]
        reconstruction_sloreta = np.zeros([data.shape[1],994,num_hfo])
                
        for hfo_index in range(num_hfo):

            individual_hfo = np.transpose(np.squeeze(data[:,:,hfo_index]))
            individual_hfo = individual_hfo / np.max(np.abs(individual_hfo))

            mid_point = int(individual_hfo.shape[0]/2);
            spike_window = individual_hfo[mid_point-128:mid_point+128,:]

            individual_hfo = highpass(individual_hfo,80,500,axis=0)

            # HF noise removal: zero out channels with HF energy above threshold
            if args.remove_hf_noise and pHFO_cells is not None and global_event_idx < len(pHFO_cells):
                phfo_ch = pHFO_cells[global_event_idx]
                if phfo_ch is not None and phfo_ch.size > 0:
                    phfo_idx = phfo_ch.flatten().astype(int) - 1
                    # Compute HF energy per channel (signal is time x channels)
                    half_win = int(0.05 * 500)
                    hf_win = individual_hfo[mid_point - half_win:mid_point + half_win, :]
                    hf_energy = np.sum(hf_win ** 2, axis=0)
                    # Threshold
                    if args.hf_threshold_mode == 'percentile':
                        threshold = np.percentile(hf_energy, args.hf_percentile)
                    elif args.hf_threshold_mode == 'median':
                        threshold = np.median(hf_energy)
                    else:
                        threshold = np.max(hf_energy[phfo_idx])
                    # Protected neighbors (shared by all pHFO channels)
                    phfo_list = phfo_idx.tolist()
                    protected_set = set(phfo_list)
                    n_protect = args.n_neighbors
                    if len(phfo_list) == 1:
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
                    noise_channels = np.where(hf_energy > threshold)[0]
                    noise_channels = np.array([ch for ch in noise_channels if ch not in protected_set])
                    if len(noise_channels) > 0:
                        individual_hfo[:, noise_channels] = 0.0
                        print(f"  Event {global_event_idx}: zeroed {len(noise_channels)} HF noise channels (mode={args.hf_threshold_mode})")

            hfo_window = individual_hfo[mid_point-50:mid_point+50,:]

            hfo_weights = np.max(np.abs(hfo_window),axis=0)/np.max(np.abs(hfo_window))
            spike_weights = np.max(np.abs(spike_window),axis=0)
          
            weights =  spike_weights
            weights = weights / np.max(weights)

            if args.suppress_distant_hfo > 0:
                
                individual_hfo =individual_hfo * weights
            individual_hfo = highpass(individual_hfo,80,500,axis=0)
            # data = data - np.mean(data, axis=0)  # Note 20211019: this only works for one source condition.
            peak_time = np.argmax(np.linalg.norm(individual_hfo, axis=1))
            # peak_time = np.argmax(d['nmm'][:,0])
            time1 = time.time()
            epoch_data = mne.EpochsArray(np.expand_dims(np.transpose(individual_hfo), 0), info, tmin=0)  #events=np.array([[436,0,0]]),
            epoch_data = epoch_data.set_eeg_reference(ref_channels='average', projection=True)
            epoch_data.apply_proj()
            data_cov = mne.compute_covariance(epoch_data, tmin=0/1000, tmax=1000/1000, method='shrunk')#
            noise_cov = mne.compute_covariance(epoch_data, tmin=0/1000, tmax=1000/1000, method='shrunk')#
            noise_cov['data'] = np.diag(np.ones(148,))  # Feb 2021: do not whitening the data
            # noise_cov['data'] = np.diag(np.diag(noise_cov['data'],))  # Feb 2021 update: do not whitening the data
            evoked = epoch_data.average()
            # ======================================= Calcualte inverse ========================================================
            # https://mne.tools/dev/auto_examples/inverse/plot_mixed_norm_inverse.html
        
            alpha = 55  # regularization parameter between 0 and 100 (100 is high)
            loose = 0.9 # loose orientation & depth weighting, original 0.9
            depth = 0.9
            n_mxne_iter = 10  # if > 1 use L0.5/L2 reweighted mixed norm solver
            # ================= LCMV results =========================================================
            time2 = time.time()
            # filters = make_lcmv(evoked.info, forward, data_cov,reg=args.reg, pick_ori=args.pick_ori, weight_norm=args.weight_norm, rank=None)  # max-power is for solution, normal is for mag
            # stc = apply_lcmv(evoked, filters)#, max_ori_out='signed')
            # lcmv_J = stc.data

            # reconstruction_lcmv[:,:,hfo_index] = np.transpose(lcmv_J)
            # --------------- perform normalization ------------------------------------------------
            # J_recon, recon_time = cal_distributed(lcmv_J, np.array([peak_time]))
            # savemat('patient_hfo/patient'+str(args.patient_id)+'/hfo_'+str(hfo_index)+'/lcmv_75.mat',{'J_recon':J_recon,'recon_time':recon_time})
            # =============== sLORETA results ======================================================
            inverse_operator = make_inverse_operator(evoked.info, forward, noise_cov, depth=depth, fixed=True, use_cps=True)
            stc_sloreta = apply_inverse(evoked, inverse_operator, lambda2=1. / 9., method='sLORETA')
            sloreta_J = stc_sloreta.data  # (20484, n_times)
            # Reduce 20484 vertices to 994 regions: one representative vertex per region (first occurrence in rm)
            _, mapping_index = np.unique(rm, return_index=True)  # (994,) indices into rm
            sloreta_994 = sloreta_J[mapping_index, :]  # (994, n_times)
            reconstruction_sloreta[:, :, hfo_index] = np.transpose(sloreta_994)
            global_event_idx += 1

            # --------------- perform normalization ------------------------------------------------
            # J_recon, recon_time = cal_distributed(sloreta_J, np.array([peak_time]),individual_hfo.shape[0])
            # savemat('patient_hfo/patient'+str(args.patient_id)+'/hfo_'+str(hfo_index)+'_sloreta_75.mat',{'J_recon':J_recon,'recon_time':recon_time})

        out_dir = 'patient_hfo/TraditionalMethod/' + args.output_folder
        os.makedirs(out_dir, exist_ok=True)
        savemat(out_dir + '/' + str(p) + '_hfo_sloreta_75.mat', {'individual_hfo': individual_hfo, 'reconstruction_sloreta': reconstruction_sloreta})
        # savemat('patient_hfo/TraditionalMethod/ssd/'+str(p)+'_hfo_lcmv_75.mat',{'reconstruction_lcmv':reconstruction_lcmv})

if __name__ == '__main__':

    mne.set_log_level('CRITICAL')
    os.environ["MKL_NUM_THREADS"] = "1"

    # ==============================  Creat info file  ==================================================================
    # ch_names = ['Fp1','F7','T7','T9','P7','O1','F3','C3','P3','Fpz','Fz','Cz','Pz','Fp2','F8','T8','T10','P8','O2','F4','C4',
    #             'P4','Nz','Iz','F9','F10','TP11','TP12','P9','P10','Oz','FT9','TP9','FT7','TP7','AF7','F5','FC5','C5','CP5',
    #             'P5','PO7','AF3','FC3','CP3','PO3','F1','FC1','C1','CP1','P1','AFz','FCz','CPz','POz','F2','FC2','C2','CP2',
    #             'P2','AF4','FC4','CP4','PO4','AF8','F6','FC6','C6','CP6','P6','PO8','FT8','TP8','FT10','TP10']
    # info = create_info(ch_names, 500, ['eeg'] * len(ch_names))
    ch_names = []
    for i in range(75):
        ch_names.append('MEG{}'.format(i))
    info = create_info(ch_names, 500, ['eeg'] * len(ch_names))

    # ==============================  Load my own forward ==================================================================
    data_path = sample.data_path()
    
    fwd_fname = str(data_path)+ "/" + '/MEG/sample/sample_audvis-eeg-oct-6-fwd.fif'
    # forward_ori = mne.read_forward_solution(fwd_fname)
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
    # forward['source_ori'] = FIFF.FIFFV_MNE_FREE_ORI
    forward['nsource'] = 20484
    forward['nchan'] = nchannel
    forward['sol']['data'] = lfd_new
    forward['sol']['nrow'] = nchannel
    forward['sol']['ncol'] = 61452
    forward['sol']['row_names'] = ch_names
    forward['_orig_sol'] = lfd_new
    forward['sol_grad'] = None
    forward['mri_head_t'] = None
    forward['info'] = info    #mne.Info(ch_names=ch_names, nchan=60)
    # Prepare source space
    src_list = [{'rr': srcmat['left_pos'], 'tris':srcmat['left_tri']-1, 'nn':srcmat['left_ori'],
                'vertno':np.arange(srcmat['left_pos'].shape[0]),'type':'surf', 'nuse':srcmat['left_idx'].size},
                {'rr': srcmat['right_pos'], 'tris':srcmat['right_tri']-1, 'nn':srcmat['right_ori'],
                'vertno':np.arange(srcmat['right_pos'].shape[0]),'type':'surf','nuse':srcmat['right_idx'].size}]

    forward['src'] = mne.SourceSpaces(src_list)
    forward['source_rr'] = np.concatenate((srcmat['left_pos'],srcmat['right_pos']))
    forward['_orig_source_ori'] = None
    forward['source_nn'] = np.tile(np.array([[1,0,0],[0,1,0],[0,0,1]]),(forward['nsource'],1))
    forward['surf_ori'] = None #True
    # ----------------------------------------------------------------------------------------------------------------------------
    global all_time_used
    all_time_used = np.zeros((2, 500))

    cal_mne(forward,srcmat)
    print('Done ', i)

    # cal_mne(603,forward,srcmat)

