import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import collections
import h5py
import hdf5storage
import torch
from torch.utils.data import DataLoader

import network as Network
import loaders as LoadDatasets
from utils import get_otsu_regions, ispadding

def pad_collate(batch):
        # Extract "data" and "nmm" from each sample in the batch
        eeg_data = [torch.tensor(sample['data'], dtype=torch.float32) for sample in batch]
        nmm_data = [torch.tensor(sample['nmm'], dtype=torch.float32) for sample in batch]
        labels = [torch.tensor(sample['label'], dtype=torch.int32) for sample in batch]
        
        # Find the maximum length in this batch
        max_len = max([seq.shape[0] for seq in eeg_data])
        
        # Initialize lists for padded sequences and masks
        padded_eeg_data, padded_nmm_data, attention_masks = [], [], []
        
        for i in range(len(batch)):
            eeg_seq, nmm_seq = eeg_data[i], nmm_data[i]
            seq_len = eeg_seq.shape[0]
            
            # Pad EEG data to max_len
            eeg_padding = torch.zeros((max_len - seq_len, eeg_seq.shape[1]))  # (padding_length, 75)
            padded_eeg_data.append(torch.cat([eeg_seq, eeg_padding], dim=0))  # (max_len, 75)
            
            # Pad NMM data to max_len
            nmm_padding = torch.zeros((max_len - seq_len, nmm_seq.shape[1]))  # (padding_length, 994)
            padded_nmm_data.append(torch.cat([nmm_seq, nmm_padding], dim=0))  # (max_len, 994)
            
            # Create mask: 1 for actual data, 0 for padding
            mask = torch.cat([torch.ones(seq_len), torch.zeros(max_len - seq_len)])
            attention_masks.append(mask)
        
        # Stack all sequences and masks for the batch
        padded_eeg_data = torch.stack(padded_eeg_data)       # (batch_size, max_len, 75)
        padded_nmm_data = torch.stack(padded_nmm_data)       # (batch_size, max_len, 994)
        attention_masks = torch.stack(attention_masks)       # (batch_size, max_len)
        label = torch.stack(labels)       # (batch_size, 70)

        return {'data': padded_eeg_data, 'nmm': padded_nmm_data, 'mask': attention_masks,'label':label}
    

def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--use_pc', type=int, default=True, metavar='t/f', help='if we are using pc for training')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--dat', default='SpikeEEGBuildEval', type=str, help='data loader')
    parser.add_argument('--test', default='test_998_snr15_extE4S1.mat', type=str, help='test dataset name')
    parser.add_argument('--model_id', type=int, default=49884, help='model id')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    parser.add_argument('--arch', default='EEGToBrainRegionTransformer', type=str, help='network')
    parser.add_argument('--num_layer', default=6, type=int, help='number of rnn layer')

    parser.add_argument('--snr_rsn_ratio', default=0, type=float, help='ratio between real noise and gaussian noise')
    parser.add_argument('--lfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')
    parser.add_argument('--hfreq', default=-1, type=int, help='filter EEG data to perform narrow-band analysis')

    parser.add_argument('--inv_lfd', default=False, type=int, help='forward model to use')
    parser.add_argument('--scale_mag', default=False, type=int, help='forward model to use')
    parser.add_argument('--scale_snr', default=False, type=int, help='forward model to use')
    parser.add_argument('--is_hfo', default=0, type=int, help='if output should be highpassed for evaluation')
    parser.add_argument('--prob_add_spike', default=0, type=float, help='if highpass the data in output')
    parser.add_argument('--highpassed_eeg', default=0, type=int, help='Frequcny at wich EEG is highpassed')
    parser.add_argument('--remove_hfo', default=0, type=int, help='if highpass the EEG')
    parser.add_argument('--variable_length', default=0, type=int, help='if highpass the EEG')
    parser.add_argument('--is_normalized', default=0, type=int, help='if normalize after highpass')
    
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = (True) and torch.cuda.is_available()  # Only use GPU during training
    device = torch.device(args.device if use_cuda else "cpu")

    if args.use_pc:
        data_root = 'source/'
        result_root = 'model_weights/{}_the_model'.format(args.model_id)
        fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd_small']
        dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']
        # if args.fwd in [ 'leadfield_20k.mat']:
        #     fwd[:, 7] = fwd[:, 994]
        #     fwd[:, 325] = fwd[:, 997]
        #     fwd[:, 921] = fwd[:, 996]
        #     fwd[:, 949] = fwd[:, 995]
        #     fwd[:, 994:] = 0
        # else:
        #     fwd = np.hstack((fwd, np.zeros((fwd.shape[0], 4))))
    else:
        data_root = 'source/'
        result_root = 'model_result/{}_the_model'.format(args.model_id)
        dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']
        fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd']  #_meg
        if args.fwd in ['leadfield_75_20k.mat', 'leadfield_20k.mat']:
            fwd[:, 7] = fwd[:, 994]
            fwd[:, 325] = fwd[:, 997]
            fwd[:, 921] = fwd[:, 996]
            fwd[:, 949] = fwd[:, 995]
            fwd[:, 994:] = 0
        else:
            fwd = np.hstack((fwd, np.zeros((fwd.shape[0], 4))))

        # fwd_t = loadmat('anatomy/leadfield_20k_conn_1_20.mat')['fwd']
        # fwd_t = np.hstack((fwd_t, np.zeros((75,4))))
    inv_ind = [439, 441, 495, 492, 498, 420, 419, 409, 408, 415, 416, 417, 418, 419, 915, 916, 918, 917, 923, 910, 922, 919]
    if args.inv_lfd:
        fwd[:, inv_ind] = -fwd[:, inv_ind]
    
    fwd = fwd[:148, :]

    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return

    # ================================== LOAD DATA ===================================================================================================
    test_data = LoadDatasets.__dict__[args.dat](data_root + args.test, fwd=fwd,
                                                args_params={'use_pc': args.use_pc, 'snr_rsn_ratio': args.snr_rsn_ratio, 'scale_snr': args.scale_snr, 'scale_mag': args.scale_mag,
                                                             'lfreq': args.lfreq, 'hfreq': args.hfreq, 'is_megeeg': False,'prob_add_spike':args.prob_add_spike, 'highpassed_eeg':args.highpassed_eeg, 'remove_hfo':args.remove_hfo,  'variable_length': args.variable_length,'is_normalized':args.is_normalized})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, collate_fn=pad_collate)
    
    data = []
    for i in range(100):
        data.append(test_data[i])
    data = np.array(data)
    savemat(result_root + '/' + args.test + '_test_100.mat',{'data':data})
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
        net = Network.__dict__[args.arch](input_channels = fwd.shape[0],num_layers=args.num_layer).to(device)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        # print("=> Loaded checkpoint {}, current results: {}".format(fn, best_result))

        # Define logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(result_root + '/outputs_eval_sim.log')
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.info("=================== Evaluation mode: {} ====================================".format(datetime.datetime.now()))
        logger.info("Testing data is {}".format(args.test))
        # Save every parameters in args
        for v in args.__dict__:
            if v not in ['workers', 'train', 'test']:
                logger.info('{} is {}'.format(v, args.__dict__[v]))
                print('{} is {}'.format(v, args.__dict__[v]))
        print('{} is {}'.format('test', args.__dict__['test']))
    else:
        print("ERROR: no checkpoint found")
        return

    # =============================== EVALUATION =====================================================================================================
    net.eval()

    eval_dict = collections.defaultdict(list)
    eval_dict['all_out'] = []                                       # DeepSIF output
    eval_dict['all_nmm'] = []                                       # Ground truth source activity
    eval_dict['all_regions'] = []                                   # DeepSIF identified source regions
    eval_dict['all_loss'] = 0                                       # MSE Loss
    criterion = torch.nn.MSELoss(reduction='sum')
    total_loss = 0
    with torch.no_grad():

        for batch_idx, sample_batch in enumerate(test_loader):
            
            
            # load data
            data = sample_batch['data'].to(device)
            nmm = sample_batch['nmm']
            label = sample_batch['label'].numpy()
            src_key_padding_mask = (sample_batch['mask'] == 0).to(device)
            
            # Transpose to fit (seq_len, batch_size, channels) expected by nn.Transformer
            data = data.transpose(0, 1)  # (max_len, batch_size, 75)
 
            
            out = net(data,src_key_padding_mask=src_key_padding_mask)
      
            # # calculate loss function
            nmm_torch = sample_batch['nmm'].to(device, torch.float)


            # Flatten arrays
            output_flat = out.reshape(-1, out.size(-1))
            nmm_flat = nmm_torch.reshape(-1, nmm_torch.size(-1))
            mask_flat = (src_key_padding_mask.reshape(-1) == 0)
            output_masked = output_flat[mask_flat]
            target_masked = nmm_flat[mask_flat]
            

            loss = criterion(output_masked, target_masked).item()
            total_loss += loss

 
            # eval_dict['all_loss'] = eval_dict['all_loss'] + loss

            # # ----- SAVE EVERYTHING TO EXAMINE LATER (not suitable for large test dataset) -------
            # eval_dict['all_out'].extend(out.cpu().numpy())
            # eval_dict['all_eeg'].extend(data.cpu().numpy())

            # ----- ONLY SAVE IDENTIFIED REGION --------------------------------------------------
                ## do not calculate metrics
            # eval_results = get_otsu_regions(out.cpu().numpy(), label)
                ## calculate metrics as a sanity check
            eval_results = get_otsu_regions(out.cpu().numpy(),nmm, label, args_params = {'dis_matrix': dis_matrix,'hfo': args.is_hfo})
            eval_dict['precision'].extend(eval_results['precision'])
            eval_dict['recall'].extend(eval_results['recall'])
            eval_dict['le'].extend(eval_results['le'])
            eval_dict['all_regions'].extend(eval_results['all_regions'])
            eval_dict['all_out'].extend(eval_results['all_out'])
            # ------------------------------------------------------------------------------------
            for kk in range(out.size(0)):
                eval_dict['all_nmm'].append(nmm[kk, :, label[kk, :, 0]].numpy())  # Only save activity in the center region
                # lb = label[kk, :, :]                                    # Save activities in all source regions
                # eval_dict['all_nmm'].append(nmm[kk, :, lb[np.logical_not(ispadding(lb))]])

    print(np.mean(eval_dict['precision']),np.mean(eval_dict['recall']),np.mean(eval_dict['le']))
    print("nan value: ",np.sum(np.isnan(eval_dict['precision'])),np.sum(np.isnan(eval_dict['recall'])),np.sum(np.isnan(eval_dict['le'])))



    # f = h5py.File(fn + '_preds_{}{}.h5'.format(args.test[:-4], args.info), 'w')
    # all_regions = np.array(eval_dict['all_regions'],dtype="object")
    # all_out = np.array(eval_dict['all_out'],dtype="object")
    # all_nmm = np.array(eval_dict['all_nmm'],dtype="object")
    # precision = np.array(eval_dict['precision'])
    # recall = np.array(eval_dict['recall'])
    # le = np.array(eval_dict['le'])
    # print(all_regions.shape)
    # print(all_out.shape)
    # print(all_nmm.shape)
    # string_dt = h5py.special_dtype(vlen=str)
    # dset = f.create_dataset("data", data =  all_regions, dtype=string_dt)
    # dset = f.create_dataset("data", data = all_out, dtype=string_dt)
    # dset = f.create_dataset("data", data = all_nmm, dtype='float32')
    # dset = f.create_dataset("data", data = precision, dtype='float32')
    # dset = f.create_dataset("data", data = recall, dtype='float32')
    # dset = f.create_dataset("data", data =  le, dtype='float32')

    hdf5storage.savemat( fn + '_preds_{}{}.mat'.format(args.test[:-4], args.info), eval_dict, format=7.3, matlab_compatible=True, compress=False )
    # savemat(fn + '_preds_{}{}.mat'.format(args.test[:-4], args.info), eval_dict,do_compression= True)

if __name__ == '__main__':
    main()



