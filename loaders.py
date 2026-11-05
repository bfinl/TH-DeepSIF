from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat, savemat
import h5py
from utils import add_white_noise, myisnan, ispadding, highpass, lowpass, detect_hfo, add_pink_noise_to_channels, bandpass
import random
import os
import sys
# import mne

class SpikeEEGBuildJR(Dataset):

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]
        if 'scale_mag' in args_params:
            self.scale_mag = args_params['scale_mag']
        else:
            self.scale_mag = 1
        if 'is_megeeg' in args_params:
            self.is_megeeg = args_params['is_megeeg']
        else:
            self.is_megeeg = 0

    def __getitem__(self, index):
        
        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(myisnan(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
            gp = self.dataset_meta['dataset_group'][index][kk].astype(np.int)
            

            true_region = curr_lb[0]
            if curr_lb[0]  == 7:
                true_region = 994
            elif curr_lb[0]  == 325:
                true_region = 997
            elif curr_lb[0]  == 921:
                true_region = 996
            elif curr_lb[0]  == 949:
                true_region = 995
            
            num_samples = len(os.listdir("source/nmm_processed_nmm/a" + str(true_region)))
           
            
            nmm_idx = random.randint(1,num_samples-20)

            current_nmm = loadmat("source/nmm_processed_nmm/a" + str(true_region) + "/nmm_" + str(nmm_idx) + ".mat")['data']
            current_nmm[:, 7] = current_nmm[:, 994]
            current_nmm[:, 325] = current_nmm[:, 997]
            current_nmm[:, 921] = current_nmm[:, 996]
            current_nmm[:, 949] = current_nmm[:, 995]
            
            
            ssig = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
            # set source space SNR
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
   
            # set weight decay inside one source patch
            if self.scale_mag:
                weight_decay = self.dataset_meta['mag_change'][0][index][kk]
                weight_decay = weight_decay[np.logical_not(myisnan(weight_decay))]
                current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay
            else:
                current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['current_snr'][index]
        noisy_eeg = add_white_noise(eeg, csnr).transpose()


        if self.is_megeeg == 2:
                       
            noisy_eeg[:, :148] = noisy_eeg[:, :148] - np.mean(noisy_eeg[:, :148], axis=0, keepdims=True)  # time
            noisy_eeg[:, :148] = noisy_eeg[:, :148]  / np.max(np.abs(noisy_eeg[:, :148] ))

            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=0, keepdims=True)
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=1, keepdims=True)  # channel
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:]  / np.max(np.abs(noisy_eeg[:, 148:]))

            # if np.random.rand(1) > 0.5:
            #      noisy_eeg[:, 148:] = - noisy_eeg[:, 148:]
        elif self.is_megeeg == 1:
            noisy_eeg[:, 0:-1:3] = noisy_eeg[:, 0:-1:3]  / np.max(np.abs(noisy_eeg[:, 0:-1:3] ))
            noisy_eeg[:, 1:-1:3] = noisy_eeg[:, 1:-1:3]  / np.max(np.abs(noisy_eeg[:, 1:-1:3] ))
            noisy_eeg[:, 2::3] = noisy_eeg[:, 2::3]  / np.max(np.abs(noisy_eeg[:, 2::3] ))

        else:
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
            noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),   #
                  'label': raw_lb,
                  'snr': csnr}
        # savemat('tmpdata{}.mat'.format(index),sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


class SpikeEEGBuild(Dataset):

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]
        if 'scale_mag' in args_params:
            self.scale_mag = args_params['scale_mag']
        else:
            self.scale_mag = 1
        if 'is_megeeg' in args_params:
            self.is_megeeg = args_params['is_megeeg']
        else:
            self.is_megeeg = 0

    def __getitem__(self, index):

        if not self.data:
           
            self.data = h5py.File('{}_nmm_all.h5'.format(self.file_path[:-12]), 'r')['data']  # TODO: CHANGE THE NAMES

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(myisnan(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
            curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
            gp = self.dataset_meta['dataset_group'][index][kk].astype(np.int)
            current_nmm = self.data['group{}'.format(gp)][self.dataset_meta['nmm_idx'][index][kk]]
            
            ssig = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
            # set source space SNR
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
   
            # set weight decay inside one source patch
            if self.scale_mag:
                weight_decay = self.dataset_meta['mag_change'][0][index][kk]
                weight_decay = weight_decay[np.logical_not(myisnan(weight_decay))]
                current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay
            else:
                current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        
        csnr = self.dataset_meta['current_snr'][index]
        noisy_eeg = add_white_noise(eeg, csnr,args_params={'is_highpassed':0}).transpose()
    


        if self.is_megeeg == 2:
                       
            noisy_eeg[:, :148] = noisy_eeg[:, :148] - np.mean(noisy_eeg[:, :148], axis=0, keepdims=True)  # time
            noisy_eeg[:, :148] = noisy_eeg[:, :148]  / np.max(np.abs(noisy_eeg[:, :148] ))

            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=0, keepdims=True)
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=1, keepdims=True)  # channel
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:]  / np.max(np.abs(noisy_eeg[:, 148:]))

            # if np.random.rand(1) > 0.5:
            #      noisy_eeg[:, 148:] = - noisy_eeg[:, 148:]
        elif self.is_megeeg == 1:
            noisy_eeg[:, 0:-1:3] = noisy_eeg[:, 0:-1:3]  / np.max(np.abs(noisy_eeg[:, 0:-1:3] ))
            noisy_eeg[:, 1:-1:3] = noisy_eeg[:, 1:-1:3]  / np.max(np.abs(noisy_eeg[:, 1:-1:3] ))
            noisy_eeg[:, 2::3] = noisy_eeg[:, 2::3]  / np.max(np.abs(noisy_eeg[:, 2::3] ))

        else:
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
            if noisy_eeg.shape[1] >1:
                noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
            noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),   #
                  'label': raw_lb,
                  'snr': csnr}
        # savemat('tmpdata{}.mat'.format(index),sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


class SpikeEEGBuildEval(Dataset):

    """Dataset, generate test data under different conditions to evaluate the model under different conditions

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging

    eval_params : dict
        New attributes compare to SpikeEEGBuild, depending on the test running, keys can be
        lfreq :         int; high pass cut-off frequency; filter EEG data to perform narrow-band analysis
        hfreq :         int; low pass cut-off frequency; filter EEG data to perform narrow-band analysis
        snr_rsn_ratio:  float; [0, 1]; ratio between real noise and gaussian noise


    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len, num_scale_ratio

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = dict()

        # check args_params:
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

        if 'snr_rsn_ratio' in args_params and args_params['snr_rsn_ratio']:                    # Need to add realistic noise
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']
        if 'lfreq' in args_params and args_params['lfreq'] > 0:
            if 'hfreq' in args_params and args_params['hfreq'] > 0:
                self.eval_params['lfreq'] = args_params['lfreq']
                self.eval_params['hfreq'] = args_params['hfreq']
            else:
                print('WARNING: NEED TO ASSIGN BOTH LOW-PASS AND HIGH-PASS CUT-OFF FREQ, IGNORE FILTERING')

    def __getitem__(self, index):

        if not self.data:
            if self.use_pc:
                self.data = h5py.File('source/test_998_snr15_nmm_all.h5', 'r')['data']
            else:
                self.data = h5py.File('{}_nmm_all.h5'.format(self.file_path[:-12]), 'r')['data']  # TODO: CHANGE THE NAMES

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        curr_lb = raw_lb[0, np.logical_not(myisnan(raw_lb[0]))]
        gp = self.dataset_meta['dataset_group'][index][0].astype(np.int)
        current_nmm_raw = self.data['group{}'.format(gp)][self.dataset_meta['nmm_idx'][index][0]]
        ssig_raw = current_nmm_raw[:, [curr_lb[0]]]                                     # waveform in the center region
   
        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
            
            curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
            current_nmm = current_nmm_raw
            # set source space SNR
            # if self.dataset_meta['scale_ratio'].shape <3:
            # ssig = ssig_raw / np.max(ssig_raw) * self.dataset_meta['scale_ratio'][index][kk]
            # else:
            ssig = ssig_raw / np.max(ssig_raw) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            weight_decay = self.dataset_meta['mag_change'][0][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['current_snr'][index]

        # add noise to sensor space
        if 'rsn' in self.eval_params:
            noisy_eeg = add_white_noise(eeg, csnr, 
                                        {'ratio': self.eval_params['snr_rsn_ratio'],
                                         'rndata': self.eval_params['rsn']['data'],
                                         'rnpower': self.eval_params['rsn']['npower']}).transpose()
        else:
            
            # noisy_eeg = add_white_noise(eeg, csnr).transpose()
            noisy_eeg = add_pink_noise_to_channels(eeg,csnr).transpose()
        # # filter data into narrow band
        # if 'lfreq' in self.eval_params:
        #     noisy_eeg = mne.filter.filter_data(np.tile(noisy_eeg.transpose(),(1,5)), 500, self.eval_params['lfreq'], self.eval_params['hfreq'],
        #                                        verbose=False).transpose()
        #     noisy_eeg = noisy_eeg[1000:1500]

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        if noisy_eeg.shape[1] >1:
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),
                  'label': raw_lb,
                  'snr': csnr}
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return sample

    def __len__(self):
        return self.dataset_len


class SpikeEEGBuildEvalJR(Dataset):

    """Dataset, generate test data under different conditions to evaluate the model under different conditions

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging

    eval_params : dict
        New attributes compare to SpikeEEGBuild, depending on the test running, keys can be
        lfreq :         int; high pass cut-off frequency; filter EEG data to perform narrow-band analysis
        hfreq :         int; low pass cut-off frequency; filter EEG data to perform narrow-band analysis
        snr_rsn_ratio:  float; [0, 1]; ratio between real noise and gaussian noise


    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len, num_scale_ratio

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = dict()

        # check args_params:
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

        if 'snr_rsn_ratio' in args_params and args_params['snr_rsn_ratio']:                    # Need to add realistic noise
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']
        if 'lfreq' in args_params and args_params['lfreq'] > 0:
            if 'hfreq' in args_params and args_params['hfreq'] > 0:
                self.eval_params['lfreq'] = args_params['lfreq']
                self.eval_params['hfreq'] = args_params['hfreq']
            else:
                print('WARNING: NEED TO ASSIGN BOTH LOW-PASS AND HIGH-PASS CUT-OFF FREQ, IGNORE FILTERING')

    def __getitem__(self, index):

        
        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]                              # labels without padding
        raw_nmm = np.zeros((500, self.fwd.shape[1]))
        
        curr_lb = raw_lb[0, np.logical_not(myisnan(raw_lb[0]))]

        true_region = curr_lb[0]
        if curr_lb[0]  == 7:
            true_region = 994
        elif curr_lb[0]  == 325:
            true_region = 997
        elif curr_lb[0]  == 921:
            true_region = 996
        elif curr_lb[0]  == 949:
            true_region = 995

        num_samples = len(os.listdir("source/nmm_processed_nmm/a" + str(true_region)))       
        nmm_idx = random.randint(num_samples-20,num_samples)

        current_nmm = loadmat("source/nmm_processed_nmm/a" + str(true_region) + "/nmm_" + str(nmm_idx) + ".mat")['data']
        current_nmm[:, 7] = current_nmm[:, 994]
        current_nmm[:, 325] = current_nmm[:, 997]
        current_nmm[:, 921] = current_nmm[:, 996]
        current_nmm[:, 949] = current_nmm[:, 995]
        
        current_nmm_raw = current_nmm
        ssig_raw = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region

        for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources

            curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
            current_nmm = current_nmm_raw
            # set source space SNR
            ssig = ssig_raw / np.max(ssig_raw) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            # set weight decay inside one source patch
            weight_decay = self.dataset_meta['mag_change'][0][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['current_snr'][index]

        # add noise to sensor space
        if 'rsn' in self.eval_params:
            noisy_eeg = add_white_noise(eeg, csnr,
                                        {'ratio': self.eval_params['snr_rsn_ratio'],
                                         'rndata': self.eval_params['rsn']['data'],
                                         'rnpower': self.eval_params['rsn']['npower']}).transpose()
        else:
            noisy_eeg = add_white_noise(eeg, csnr).transpose()

        # # filter data into narrow band
        # if 'lfreq' in self.eval_params:
        #     noisy_eeg = mne.filter.filter_data(np.tile(noisy_eeg.transpose(),(1,5)), 500, self.eval_params['lfreq'], self.eval_params['hfreq'],
        #                                        verbose=False).transpose()
        #     noisy_eeg = noisy_eeg[1000:1500]

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),
                  'label': raw_lb,
                  'snr': csnr}
        if self.transform:
            sample = self.transform(sample)

        # savemat('{}/data{}.mat'.format(self.file_path[0][:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb]})
        return sample

    def __len__(self):
        return self.dataset_len



class SpikeNMMDatah5SZTMP3(Dataset):

    def __init__(self, data_root, fwd=np.array([0]), transform=None, args_params={}):

        random.seed(15213)

        self.file_path = data_root
        self.dataset_len = []
        self.num_scale = []
        self.fwd = fwd
        self.transform = transform
        self.use_pc = args_params['use_pc']
        self.scale_snr = args_params['scale_snr']
        self.scale_mag = args_params['scale_mag']
        self.snr_rsn_ratio = args_params['snr_rsn_ratio']

        self.dataset_meta = loadmat(self.file_path)
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]
        if 'scale_mag' in args_params:
            self.scale_mag = args_params['scale_mag']
        else:
            self.scale_mag = 1
        if 'is_megeeg' in args_params:
            self.is_megeeg = args_params['is_megeeg']
        else:
            self.is_megeeg = 0

        self.rsn = loadmat('anatomy/realistic_noise.mat')
        self.dataset_meta['sz_source'] = loadmat('anatomy/ictal_source_waveform_new.mat')['source']
        self.data = []
        self.selected_region = []
        self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if self.scale_snr:
            self.num_scale = self.dataset_meta['scale_ratio'].shape[2]
        if self.scale_mag:
            self.num_mag = self.dataset_meta['mag_change'].shape[0]

    def __getitem__(self, index):

        if not self.data:
        
            self.data = h5py.File('source/test_998_snr15_nmm_all.h5','r')['data']  # TODO: CHANGE THE NAMES

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)
        raw_nmm = np.zeros((500, self.fwd.shape[1]))
        lb = raw_lb[np.logical_not(myisnan(raw_lb))]
        # lb = raw_lb[0][np.logical_not(myisnan(raw_lb[0]))]
        ## Try training it with only one sources. ---------------------
        for kk in range(raw_lb.shape[0]):  # number of sources  #raw_lb.shape[0]
        # for kk in range(2):  # number of sources
            tmplb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
            gp = self.dataset_meta['dataset_group'][index][kk].astype(np.int)
            current_nmm = self.data['group{}'.format(gp)][self.dataset_meta['nmm_idx'][index][kk]]
            noise_sig_candidate = np.hstack((current_nmm[:, :tmplb[0]], current_nmm[:, tmplb[0] + 1:]))
            noise_sig_candidate=noise_sig_candidate-np.mean(noise_sig_candidate)
            tmp_idx=list(range(noise_sig_candidate.shape[1]))
            tmp_idx=np.random.permutation(tmp_idx)
            noise_sig_candidate=np.reshape(noise_sig_candidate[:,tmp_idx[:len(tmplb)]], [1, 500, -1])

            ssig = current_nmm[:, [tmplb[0]]]
            if self.scale_snr:
                ssig = ssig / np.max(abs(ssig)) * self.dataset_meta['scale_ratio'][index][kk][
                    random.randint(0, self.num_scale - 1)]
            else:
                ssig = ssig / (min([0.8 * len(tmplb), 10]) * 1)
            if 'sz_source' in self.dataset_meta:
                sig_type = random.randint(0, 3) * 5000
                type_ind = [random.randint(0, 4999) + sig_type, random.randint(0, 4999) + sig_type]
                ssig = self.dataset_meta['sz_source'][type_ind] * np.max(ssig)  # 'sz_source' is normalized to be 1
                # ssig = np.tile(self.dataset_meta['source'][:, kk] * 500, (len(tmplb), 1)).transpose() / min([0.8 * len(tmplb), 5.])
            if random.random() < 2:
                same_source_ind = random.randint(0, 10)
            else:
                same_source_ind = len(tmplb)
            if self.scale_mag:
                vv = self.dataset_meta['mag_change'][random.randint(0, self.num_mag - 1)][index][kk][:]  #
                vv = vv[np.logical_not(myisnan(vv))]
                current_nmm[:, tmplb[0:same_source_ind]] = ssig[0, :].reshape(-1, 1) * vv[0:same_source_ind]
                current_nmm[:, tmplb[same_source_ind:]] = ssig[1, :].reshape(-1, 1) * vv[same_source_ind:]
            else:
                current_nmm[:, tmplb[0:same_source_ind]] = ssig[0, :].reshape(-1, 1)
                current_nmm[:, tmplb[same_source_ind:]] = ssig[1, :].reshape(-1, 1)

            # current_nmm[:, tmplb]=add_white_noise(current_nmm[:, tmplb].transpose(), 10, 1.0, noise_sig_candidate, np.square(np.linalg.norm(noise_sig_candidate, axis=1)) / noise_sig_candidate.shape[0]).transpose()
            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())
        csnr = self.dataset_meta['current_snr'][index]
        noisy_eeg = add_white_noise(eeg, csnr).transpose()
        # scale the eeg
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))
        # get the nmm
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]

        # empty_nmm = empty_nmm / empty_nmm.max()
        empty_nmm = empty_nmm / np.abs(empty_nmm).max()
        empty_nmm = empty_nmm[:, :994]
        # Save each data
        # savemat('D:/Database/MySim/NMM_DL/source/data{}.mat'.format(index),
        #         {'data': noisy_eeg, 'nmm': empty_nmm[:, lb]})
        bw_label = np.zeros(994)
        bw_label[lb] = 1
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),
                  'label': raw_lb,
                  'bw_label': bw_label.astype('float32'),
                  'snr': csnr}
        if self.transform:
            sample = self.transform(sample)
        # savemat('{}_tmp/data{}.mat'.format(self.file_path[:-4],index),{'data':noisy_eeg,'label':raw_lb,'nmm':empty_nmm[:,lb],'r': same_source_ind})
        # savemat('source/data{}_{}.mat'.format(self.file_path[0][:-4], index),
        #         {'data': noisy_eeg, 'label': raw_lb, 'nmm': empty_nmm[:, lb]})
        return sample

    def __len__(self):
        return self.dataset_len
        
# from matplotlib import pyplot as plt
# plt.subplot(1,2,1)
# plt.plot(noisy_eeg)
# plt.subplot(1,2,2)
# plt.plot(empty_nmm[:,lb])

class HFOEEGBuildJR(Dataset):

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        self.data = []
        self.data_hfo = []
        
        self.dataset_meta = loadmat(self.file_path)
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]
        if 'scale_mag' in args_params:
            self.scale_mag = args_params['scale_mag']
        else:
            self.scale_mag = 1
        if 'is_megeeg' in args_params:
            self.is_megeeg = args_params['is_megeeg']
        else:
            self.is_megeeg = 0
        if 'is_highpassed' in args_params:
            self.is_highpassed = args_params['is_highpassed']
        else:
            self.is_highpassed = False
        if 'highpassed_eeg' in args_params:
            self.highpassed_eeg = args_params['highpassed_eeg']
        else:
            self.highpassed_EEG = 0
        if 'prob_add_spike' in args_params:
            self.prob_add_spike = args_params['prob_add_spike']
        else:
            self.prob_add_spike = 0
        if 'is_normalized' in args_params:
            self.is_normalized = args_params['is_normalized']
        else:
            self.is_normalized = 0
        if 'prob_remove_hfo' in args_params:
            self.prob_remove_hfo = args_params['prob_remove_hfo']
        if 'variable_length'in args_params:
            self.variable_length = args_params['variable_length']
        else:
            self.variable_length = 0
   
            


    def __getitem__(self, index):

        # try:
        #     with h5py.File("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat", 'r') as f:
        #         raw_nmm = f['nmm']  # Access a specific variable
        # except OSError:
        #     try:
        #         raw_nmm = loadmat("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat")['nmm']
        #     except Exception as e:
        #         print(f"Failed to load {index+1} with both methods: {e}")

        try:
            with h5py.File("hfo_source/hfo_complete_dataset_train_noise001002_cf90110_snr2050/"+str(index+1)+".mat", 'r') as f:
                # For v7.3 format, the data is stored in a specific path structure
                if 'final_signal' in f:
                    raw_nmm = f['final_signal'][:]
                else:
                    # Try the v7.3 path structure - variables are stored in #refs#
                    if '#refs#' in f and 'final_signal' in f['#refs#']:
                        raw_nmm = f['#refs#']['final_signal'][:]
                    else:
                        # Try to find the variable in the root or other common locations
                        found = False
                        for key in f.keys():
                            if key == 'final_signal':
                                raw_nmm = f[key][:]
                                found = True
                                break
                        if not found:
                            raise KeyError("final_signal not found in any expected location")
        except (OSError, ValueError, KeyError) as e:
            print(f"Failed to load with h5py: {e}")
            try:
                # Try with squeeze_me=True for better v7.3 compatibility
                mat_data = loadmat("hfo_source/hfo_complete_dataset_train_noise001002_cf90110_snr2050/"+str(index+1)+".mat", squeeze_me=True)
                raw_nmm = mat_data['final_signal']
            except Exception as e:
                print(f"Failed to load {index+1} with both methods: {e}")
        # print("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat")
        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int)         # labels with padding
        lb = raw_lb[np.logical_not(myisnan(raw_lb))]
        raw_nmm = raw_nmm.transpose();
        if np.isnan(raw_nmm).sum()>0:

            print("Nan in original nmm import: ", index)

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        # empty_nmm = raw_nmm
        if self.is_highpassed:
            empty_nmm = highpass(empty_nmm,80,500,axis=0)
            empty_nmm = empty_nmm[25:-25,:]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]

        
        power_total = np.sum(np.square(np.linalg.norm(raw_nmm,axis=0))/raw_nmm.shape[0])  # Calculate the signal power including HFO

        
        if random.random() < self.prob_add_spike:

            if not self.data:
                
                if self.file_path[7:12] == 'train': 
                    
                    self.data = h5py.File('source/train_998_snr15_nmm_all.h5', 'r')['data']  # TODO: CHANGE THE NAMES

                else:
           
                    self.data = h5py.File('source/test_998_snr15_nmm_all.h5', 'r')['data']  # TODO: CHANGE THE NAMES
  
            for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
                
                curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
                gp = self.dataset_meta['dataset_group'][index][0].astype(np.int)
                current_nmm = self.data['group{}'.format(gp)][self.dataset_meta['nmm_idx'][index][kk]]
                ssig_raw = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
                
                spike_to_background_snr = random.randint(0,5)
                power_background = power_total/994# * (994-len(curr_lb))/994
                target_power_signal = 10 ** ((spike_to_background_snr / 10)) * power_background
            
                # set source space SNR
                power_signal = np.square(np.linalg.norm(ssig_raw,axis=0))/ssig_raw.shape[0]

                if not power_signal == 0:
                    ssig_raw = ssig_raw.reshape(-1, 1)*np.sqrt(target_power_signal/power_signal)
                    # set weight decay inside one source patch
                    weight_decay = self.dataset_meta['mag_change'][0][index][kk]
                    weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
                    current_nmm[:, curr_lb] = ssig_raw * weight_decay
            
                    raw_nmm[:, curr_lb] = raw_nmm[:, curr_lb] + current_nmm[:, curr_lb] 

        eeg = np.matmul(self.fwd, raw_nmm.transpose())                              # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['current_snr'][index]
        # noisy_eeg = add_white_noise(eeg, csnr,args_params={'is_highpassed':0}).transpose()
        noisy_eeg = add_pink_noise_to_channels(eeg,csnr).transpose()



        if self.highpassed_eeg > 0 and self.is_normalized:
            noisy_eeg = highpass(noisy_eeg,self.highpassed_eeg,500,axis=0)
            noisy_eeg = noisy_eeg[25:-25,:]

        if np.isnan(noisy_eeg).sum()>0:
            print("EEG contain Nan in loader after adding spike: ",index)
        if np.isnan(empty_nmm).sum()>0:
            print("NMM contain Nan in loader after adding spike: ",index)

        if self.is_megeeg == 2:
                       
            noisy_eeg[:, :148] = noisy_eeg[:, :148] - np.mean(noisy_eeg[:, :148], axis=0, keepdims=True)  # time
            noisy_eeg[:, :148] = noisy_eeg[:, :148]  / np.max(np.abs(noisy_eeg[:, :148] ))

            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=0, keepdims=True)
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:] - np.mean(noisy_eeg[:, 148:], axis=1, keepdims=True)  # channel
            noisy_eeg[:, 148:] = noisy_eeg[:, 148:]  / np.max(np.abs(noisy_eeg[:, 148:]))

            # if np.random.rand(1) > 0.5:
            #      noisy_eeg[:, 148:] = - noisy_eeg[:, 148:]
        elif self.is_megeeg == 1:
            noisy_eeg[:, 0:-1:3] = noisy_eeg[:, 0:-1:3]  / np.max(np.abs(noisy_eeg[:, 0:-1:3] ))
            noisy_eeg[:, 1:-1:3] = noisy_eeg[:, 1:-1:3]  / np.max(np.abs(noisy_eeg[:, 1:-1:3] ))
            noisy_eeg[:, 2::3] = noisy_eeg[:, 2::3]  / np.max(np.abs(noisy_eeg[:, 2::3] ))

        else:
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
            noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
            noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        if self.highpassed_eeg > 0 and not self.is_normalized:
            noisy_eeg = highpass(noisy_eeg,self.highpassed_eeg,500,axis=0)
            noisy_eeg = noisy_eeg[25:-25,:]


        # Segment the data for variable length

        if self.variable_length:

            hfo_segments, envelope, threshold = detect_hfo(empty_nmm[:,lb[0]])
            if len(hfo_segments) == 0:
                hfo_start = 100
            else:
                hfo_start = hfo_segments[0][0]

            data_length = random.randint(100,500)
            before_hfo = random.randint(20,data_length-80)
            segment_start = hfo_start - before_hfo
            
            if  segment_start< 0:
                noisy_eeg = noisy_eeg[0:data_length,:]
                empty_nmm = empty_nmm[0:data_length,:]
            else:
                noisy_eeg = noisy_eeg[segment_start:segment_start+data_length,:]
                empty_nmm = empty_nmm[segment_start:segment_start+data_length,:]

        # Check Nan
        if np.isnan(noisy_eeg).sum()>0:
            print("EEG contain Nan in loader final: ",index)
        if np.isnan(empty_nmm).sum()>0:
            print("NMM contain Nan in loader final: ",index)
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),   #
                  'label': raw_lb,
                  'snr': csnr}
        # savemat('tmpdata{}.mat'.format(index),sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len


class HFOEEGBuildEvalJR(Dataset):

    """Dataset, generate test data under different conditions to evaluate the model under different conditions

    Attributes
    ----------
    data_root : str
        Dataset file location
    fwd : np.array
        Size is num_electrode * num_region
    data : np.array
        TVB output data
    dataset_meta : dict
        Information needed to generate data
        selected_region: spatial model for the sources; num_examples * num_sources * max_size
                         num_examples: num_examples in this dataset
                         num_sources: num_sources in one example
                         max_size: cortical regions in one source patch; first value is the center region id; variable length, padded to max_size
                            (set to 70, an arbitrary number)
        nmm_idx:         num_examples * num_sources: index of the TVB data to use as the source
        scale_ratio:     scale the waveform maginitude in source region; num_examples * num_sources * num_scale_ratio (num_snr_level)
        mag_change:      magnitude changes inside a source patch; num_examples * num_sources * max_size
                         weight decay inside a patch; equals to 1 in the center region; variable length; padded to max_size
        sensor_snr:      the Gaussian noise added to the sensor space; num_examples * 1;

    dataset_len : int
        size of the dataset, can be set as a small value during debugging

    eval_params : dict
        New attributes compare to SpikeEEGBuild, depending on the test running, keys can be
        lfreq :         int; high pass cut-off frequency; filter EEG data to perform narrow-band analysis
        hfreq :         int; low pass cut-off frequency; filter EEG data to perform narrow-band analysis
        snr_rsn_ratio:  float; [0, 1]; ratio between real noise and gaussian noise


    """

    def __init__(self, data_root, fwd, transform=None, args_params=None):

        # args_params: optional parameters; can be dataset_len, num_scale_ratio

        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = dict()

        # check args_params:
        self.use_pc = args_params['use_pc']
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:   # use the whole dataset
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

        if 'snr_rsn_ratio' in args_params and args_params['snr_rsn_ratio']:                    # Need to add realistic noise
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']
        if 'lfreq' in args_params and args_params['lfreq'] > 0:
            if 'hfreq' in args_params and args_params['hfreq'] > 0:
                self.eval_params['lfreq'] = args_params['lfreq']
                self.eval_params['hfreq'] = args_params['hfreq']
            else:
                print('WARNING: NEED TO ASSIGN BOTH LOW-PASS AND HIGH-PASS CUT-OFF FREQ, IGNORE FILTERING')
        if 'prob_add_spike' in args_params:
            self.prob_add_spike = args_params['prob_add_spike']
        else:
            self.prob_add_spike = 0
        if 'highpassed_eeg' in args_params:
            self.highpassed_eeg = args_params['highpassed_eeg']
        else:
            self.highpassed_EEG = 0
        if 'remove_hfo' in args_params:
            self.remove_hfo = args_params['remove_hfo']
            print("self.remove_hfo",self.remove_hfo)
        else:
            self.remove_hfo = 0
        if 'is_normalized' in args_params:
            self.is_normalized = args_params['is_normalized']
        else:
            self.is_normalized = 0
        if 'variable_length'in args_params:
            self.variable_length = args_params['variable_length']
        else:
            self.variable_length = 0

    def __getitem__(self, index):

        try:
            with h5py.File("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat", 'r') as f:
                raw_nmm = f['nmm']  # Access a specific variable
        except OSError:
            try:
                raw_nmm = loadmat("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat")['nmm']
            except Exception as e:
                print(f"Failed to load {index+1} with both methods: {e}")
        # print("hfo_source/"+self.file_path[7:-4]+"_10012024/"+str(index+1)+".mat")
        raw_lb = self.dataset_meta['selected_region'][index].astype(int)         # labels with padding
        lb = raw_lb[np.logical_not(myisnan(raw_lb))]
        if np.isnan(raw_nmm).sum()>0:

            print("Nan in original nmm import: ", index)

        # get the training output
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        # empty_nmm = raw_nmm
        
        empty_nmm = bandpass(empty_nmm,80,240,500,axis=0)
        empty_nmm = empty_nmm[25:-25,:]
        empty_nmm = empty_nmm / np.max(empty_nmm)
        empty_nmm = empty_nmm[:, :994]

        
        power_total = np.sum(np.square(np.linalg.norm(raw_nmm,axis=0))/raw_nmm.shape[0])  # Calculate the signal power including HFO

        
        if random.random() < self.prob_add_spike:

            if not self.data:
                
                if self.file_path[7:12] == 'train': 
                    
                    self.data = h5py.File('source/train_998_snr15_nmm_all.h5', 'r')['data']  # TODO: CHANGE THE NAMES

                else:
           
                    self.data = h5py.File('source/test_998_snr15_nmm_all.h5', 'r')['data']  # TODO: CHANGE THE NAMES
  
            for kk in range(raw_lb.shape[0]):                                           # iterate through number of sources
                
                curr_lb = raw_lb[kk, np.logical_not(myisnan(raw_lb[kk]))]
                gp = self.dataset_meta['dataset_group'][index][0].astype(int)
                current_nmm = self.data['group{}'.format(gp)][self.dataset_meta['nmm_idx'][index][kk]]
                ssig_raw = current_nmm[:, [curr_lb[0]]]                                     # waveform in the center region
                
                spike_to_background_snr = random.randint(0,5)
                power_background = power_total/994# * (994-len(curr_lb))/994
                target_power_signal = 10 ** ((spike_to_background_snr / 10)) * power_background
            
                # set source space SNR
                power_signal = np.square(np.linalg.norm(ssig_raw,axis=0))/ssig_raw.shape[0]

                if not power_signal == 0:
                    ssig_raw = ssig_raw.reshape(-1, 1)*np.sqrt(target_power_signal/power_signal)
                    # set weight decay inside one source patch
                    weight_decay = self.dataset_meta['mag_change'][0][index][kk]
                    weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
                    current_nmm[:, curr_lb] = ssig_raw * weight_decay
            
                    raw_nmm[:, curr_lb] = raw_nmm[:, curr_lb] + current_nmm[:, curr_lb] 

        eeg = np.matmul(self.fwd, raw_nmm.transpose())  
                                    # project data to sensor space; num_electrode * num_time
        csnr = self.dataset_meta['current_snr'][index]
        # noisy_eeg = add_white_noise(eeg, csnr,args_params={'is_highpassed':0}).transpose()
        noisy_eeg = add_pink_noise_to_channels(eeg,csnr).transpose()



        if self.highpassed_eeg > 0 and self.is_normalized:
            noisy_eeg = bandpass(noisy_eeg,self.highpassed_eeg,240,500,axis=0)
            noisy_eeg = noisy_eeg[25:-25,:]

        if np.isnan(noisy_eeg).sum()>0:
            print("EEG contain Nan in loader after adding spike: ",index)
        if np.isnan(empty_nmm).sum()>0:
            print("NMM contain Nan in loader after adding spike: ",index)

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)  # time
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)  # channel
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        if self.highpassed_eeg > 0 and not self.is_normalized:
            noisy_eeg = bandpass(noisy_eeg,self.highpassed_eeg,240,500,axis=0)
            noisy_eeg = noisy_eeg[25:-25,:]


        # Segment the data for variable length

        if self.variable_length:

            hfo_segments, envelope, threshold = detect_hfo(empty_nmm[:,lb[0]])
            if len(hfo_segments) == 0:
                hfo_start =0
            else:
                hfo_start = hfo_segments[0][0]

            data_length = random.randint(100,500)
            before_hfo = random.randint(20,data_length-80)
            segment_start = hfo_start - before_hfo
            
            if  segment_start< 0:
                noisy_eeg = noisy_eeg[0:data_length,:]
                empty_nmm = empty_nmm[0:data_length,:]
            else:
                noisy_eeg = noisy_eeg[segment_start:segment_start+data_length,:]
                empty_nmm = empty_nmm[segment_start:segment_start+data_length,:]

        # Check Nan
        if np.isnan(noisy_eeg).sum()>0:
            print("EEG contain Nan in loader final: ",index)
        if np.isnan(empty_nmm).sum()>0:
            print("NMM contain Nan in loader final: ",index)
        # Each data sample
        sample = {'data': noisy_eeg.astype('float32'),
                  'nmm': empty_nmm.astype('float32'),   #
                  'label': raw_lb,
                  'snr': csnr}
        # savemat('tmpdata{}.mat'.format(index),sample)
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len
