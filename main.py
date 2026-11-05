import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import sys

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader



import network as Network

import loaders as LoadDatasets
from utils import myisnan, highpass

# from utils import add_white_noise, myisnan, ispadding, highpass, lowpass,detect_hfo

# 05042022 Clean up the main function, remove eval from it

def main_train():

    print("number of cpu:",os.cpu_count())
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='Rachel DLSL Model')
    parser.add_argument('--use_pc', type=int, default=True, metavar='t/f', help='if we are using pc for training')
    parser.add_argument('--save', type=int, default=True, metavar='t/f', help='save the model or not')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=64, type=int, metavar='bs', help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, metavar='bs', help='batch size')

    # --------- Parameters to check before every run ----------------------------
    parser.add_argument('--arch', default='EnahncedEEGToBrainRegionTransformer', type=str, help='network')
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str, help='dataset type')
    parser.add_argument('--train', default='test_998_snr15_extenL2.mat', type=str, help='train dataset name')
    parser.add_argument('--test', default='test_998_snr15_extenL2.mat', type=str, help='test dataset name')
    parser.add_argument('--model_id', default=0, type=int, help='model id')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--epoch', default=50, type=int, help='total number of epoch')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    parser.add_argument('--num_layer', default=6, type=int, help='number of rnn layer')
    parser.add_argument('--drop_out', default=0.2, type=float, help='dropout rate')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    
    parser.add_argument('--eval', default=False, type=int, help='Only run the evaluation for the model')
    parser.add_argument('--loss_weighting', default=0, type=float, help='Loss weighting for the model')
    parser.add_argument('--scale_snr', default=False, type=int, help='forward model to use')
    parser.add_argument('--scale_mag', default=False, type=int, help='forward model to use')
    parser.add_argument('--inv_lfd', default=False, type=int, help='forward model to use')
    parser.add_argument('--pos_weight', default=False, type=int, help='forward model to use')
    parser.add_argument('--snr_rsn_ratio', default=0, type=float, help='forward model to use')
    parser.add_argument('--is_highpassed', default=0, type=int, help='if highpass the data in output')
    parser.add_argument('--highpassed_eeg', default=0, type=int, help='if highpass the EEG')
    parser.add_argument('--prob_add_spike', default=0, type=float, help='probability of adding spike to souces')
    parser.add_argument('--prob_remove_hfo', default=0, type=float, help='probability of removing hfo activity in souce')
    parser.add_argument('--is_normalized', default=0, type=int, help='if normalize after highpass')
    parser.add_argument('--variable_length', default=0, type=int, help='Cut data sample to different legnth for model generalization')

    args = parser.parse_args()

    # ------------------------ Prepare ----------------------------------
    if args.use_pc:  # to save time when run on the server
        from matplotlib import pyplot as plt
    # Predefined Parameters
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    # cudnn.benchmark = True

    if args.use_pc:
        data_root = 'source/'
        result_root = 'model_weights/{}_the_model'.format(args.model_id)
        fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd_small']
        # if fwd.shape[1] == 998:
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
        fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd']
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
            # fwd_t[:,inv_ind] = -fwd_t[:,inv_ind]
        # fwd = fwd[:148, :]
        # fwd_list = []
        # fwd_list.append(fwd)
        # fwd_list.append(fwd_t)
        # fwd = fwd_list

    if not os.path.exists(result_root):
        os.makedirs(result_root)


    # Define logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(result_root + '/outputs_{}.log'.format(args.arch))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("====================={}====================================".format(datetime.datetime.now()))
    logger.info("This is eval: {}".format(args.eval))
    logger.info("Training data is {}, and testing data is {}".format(args.train, args.test))
    

    # ================================== LOAD DATA ==========================================================================
    train_data = LoadDatasets.__dict__[args.dat](data_root + args.train, fwd=fwd,
                                                    args_params={'use_pc': args.use_pc, 'is_megeeg':0,'snr_rsn_ratio':args.snr_rsn_ratio,'scale_snr':args.scale_snr,'scale_mag':args.scale_mag,'is_highpassed':args.is_highpassed,'prob_add_spike':args.prob_add_spike,'prob_remove_hfo':args.prob_remove_hfo, 'highpassed_eeg':args.highpassed_eeg, 'variable_length': args.variable_length,'is_normalized':args.is_normalized})
  
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=pad_collate)
    data = []
    for i in range(100):
        data.append(train_data[i])
    data = np.array(data)
    savemat(result_root + '/train_100.mat',{'data':data})
    test_data = LoadDatasets.__dict__[args.dat](data_root + args.test, fwd=fwd,
                                                args_params={'use_pc': args.use_pc, 'is_megeeg':0,'snr_rsn_ratio':args.snr_rsn_ratio,'scale_snr':args.scale_snr,'scale_mag':args.scale_mag,'is_highpassed':args.is_highpassed,'prob_add_spike':args.prob_add_spike,'prob_remove_hfo':args.prob_remove_hfo,'highpassed_eeg':args.highpassed_eeg, 'variable_length': args.variable_length,'is_normalized':args.is_normalized})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False, collate_fn=pad_collate)
  
    # ================================== CREATE MODEL ====================================================================

    
    net = Network.__dict__[args.arch](input_channels = fwd.shape[0],num_layers=args.num_layer, dropout=args.drop_out).to(device)
    
    # torch.cuda.memory_summary()
    # torch.cuda.memory_allocated()
    # checkpoint = torch.load('model_result/10284_the_model/model_best.pth.tar', map_location=torch.device('cpu'))   
    # net = Network.__dict__[checkpoint['arch']](*checkpoint['attribute_list']).to(device)  #[:-1]
    # net.load_state_dict(checkpoint['state_dict'], strict=False)
    # print("=> Loaded previous model")

    optimizer_s = 'optim.Adam(net.parameters(), lr={}, weight_decay=1e-6)'.format(args.lr)
    optimizer = eval(optimizer_s)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2,
    #                                                     patience=5, verbose=True, threshold=0.001)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    criterion = nn.MSELoss(reduction='sum')

    # Print every parameters in args
    for v in args.__dict__:
        
        logger.info('{} is {}'.format(v, args.__dict__[v]))

    args.start_epoch = 0
    best_result = np.Inf
    train_loss = []
    test_loss = []
    # =============================== RESUME ============================================================================
    if args.resume:
        print(result_root)
        fn = os.path.join(result_root, 'epoch_' + args.resume)
        if os.path.isfile(fn):
            print("=> Found checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(fn, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            # redefine the weights architecture
            # net = Network.__dict__[args.arch](*checkpoint['attribute_list']).to(device)
            # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
            net = Network.__dict__[args.arch](input_channels = fwd.shape[0],num_layers=args.num_layer, dropout=args.drop_out).to(device)
            optimizer = eval(checkpoint['optimizer_s'])
            net.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = param_group['lr'] / checkpoint['lr'] * args.lr
            print("=> loaded checkpoint epoch {}, best results: {}"
                  .format(args.resume, best_result))
            sts = loadmat(result_root + '/train_test_error.mat')
            train_loss.extend(sts['train_loss'][0][:int(args.resume) + 1])
            test_loss.extend(sts['test_loss'][0][:int(args.resume) + 1])
            no_load_model = 0
        else:
            print("=> no checkpoint found at '{}', use random weights".format(args.resume))

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # ------------------ start training/evaluation process -----------------------------------
    for epoch in range(args.start_epoch + 1, args.epoch):


        # adjust_learning_rate
        # lr = args.lr * (0.1 ** (epoch // 20))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr

        # train for one epoch
        # train for one epoch
        train_lss_all = train(train_loader, net, criterion, optimizer, {'fwd':fwd[:,:994],'device': device, 'logger': logger, 'epoch': epoch,'profiler': result_root})
        # evaluate on validation set
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        scheduler.step()
        train_loss.extend([np.sum(np.array(train_lss_all)) / len(train_data)])
        test_loss.extend([np.sum(np.array(test_lss_all))/len(test_data)])

        print_s = 'Epoch {}: Time:{:6.2f}, '.format(epoch, time.time() - start_time) + \
                  'Train Loss:{:06.5f}'.format(train_loss[-1]) + ', Test Loss:{:06.5f}'.format(test_loss[-1])
        print(print_s)
        logger.info(print_s)
        is_best = test_loss[-1] < best_result
        best_result = min(test_loss[-1], best_result)
        if is_best:
            torch.save({
                'epoch': epoch, 'arch': args.arch, 'state_dict': net.state_dict(),
                'best_result': best_result, 'lr': args.lr, 'info': args.info,
                'train': args.train, 'test': args.test,
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(), 'optimizer_s': optimizer_s},
                result_root + '/model_best.pth.tar')
        if args.save:
            # save checkpoint
            torch.save({
                'epoch': epoch, 'arch': args.arch, 'state_dict': net.state_dict(),
                'best_result': best_result, 'lr': args.lr, 'info': args.info,
                'train': args.train, 'test': args.test, 
                'scheduler_state_dict': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(), 'optimizer_s': optimizer_s},
                result_root + '/epoch_{}'.format(epoch))
            savemat(result_root + '/train_test_error.mat', {'train_loss': train_loss, 'test_loss': test_loss})
            savemat(result_root + '/train_test_loss_epoch{}.mat'.format(epoch), {'train_loss': train_lss_all, 'test_loss': test_lss_all})
    # END MAIN_TRAIN




def train(train_loader, model, criterion, optimizer, args_params):
    # args_params: potential parameter inputs, could be "device","loss_weighting"
    # exec has variable scope issue
    device = args_params['device']
    logger = args_params['logger']
    if 'lw' in args_params: lw = args_params['lw']
    # switch to train mode
    model.train()
    train_loss = 0
    start_time = time.time()

    with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(args_params['profiler']),
    with_stack=True
    ) as profiler:
        for batch_idx, sample_batch in enumerate(train_loader):

            # load data
            eeg_data = sample_batch['data'].to(device)
            target_nmm = sample_batch['nmm'].to(device)
            src_key_padding_mask = (sample_batch['mask'] == 0).to(device)
            
            # Transpose to fit (seq_len, batch_size, channels) expected by nn.Transformer
            eeg_data = eeg_data.transpose(0, 1)  # (max_len, batch_size, 75)
            target_nmm = target_nmm.transpose(0, 1)  # (max_len, batch_size, 994)

            # Forward pass with masking
            optimizer.zero_grad(set_to_none=True)
            output = model(eeg_data, src_key_padding_mask=src_key_padding_mask)
            

            # Flatten outputs and targets to (batch_size * seq_length, brain_regions)
            output_flat = output.reshape(-1, output.size(-1))
            target_nmm = target_nmm.transpose(0, 1)
            target_flat = target_nmm.reshape(-1, target_nmm.size(-1))
            
            # Flatten and invert mask to match the flattened shape
            mask_flat = (src_key_padding_mask.reshape(-1) == 0)
            
            # Apply mask: select only the unpadded elements
            output_masked = output_flat[mask_flat]
            target_masked = target_flat[mask_flat]
            
            # Calculate MSE loss on masked elements
            loss = criterion(output_masked, target_masked)

            loss.backward()
            optimizer.step()
            profiler.step()

            train_loss += loss.item()
            if (batch_idx + 1) % 500 == 0:
                print_s = "batch_idx_{}_time_{}_train_loss_{}".format(batch_idx, time.time() - start_time, train_loss)
                logger.info(print_s)
                
                # torch.save({
                #     'state_dict': model.state_dict(), 'attribute_list': model.attribute_list},
                #     'model_result/148846_the_model/epoch_{}_batch_{}'.format(args_params['epoch'], batch_idx))

    # train_loss = torch.cat(train_loss).cpu().view(-1, nloss).numpy()

    return train_loss

# START VALIDATE FUNC
def validate(val_loader, model, criterion, args_params):
    # switch to evaluate mode
    device = args_params['device']

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(val_loader):

            # load data
            eeg_data = sample_batch['data'].to(device)
            target_nmm = sample_batch['nmm'].to(device)
            src_key_padding_mask = (sample_batch['mask'] == 0).to(device)

            # Transpose to fit (seq_len, batch_size, channels) expected by nn.Transformer
            eeg_data = eeg_data.transpose(0, 1)  # (max_len, batch_size, 75)
            target_nmm = target_nmm.transpose(0, 1)  # (max_len, batch_size, 994)

            output = model(eeg_data, src_key_padding_mask=src_key_padding_mask)
            
            # Flatten outputs and targets to (batch_size * seq_length, brain_regions)
            output_flat = output.reshape(-1, output.size(-1))
            target_nmm = target_nmm.transpose(0, 1)
            target_flat = target_nmm.reshape(-1, target_nmm.size(-1))
            
            
            # Flatten and invert mask to match the flattened shape
            mask_flat = (src_key_padding_mask.reshape(-1) == 0)
            
            # Apply mask: select only the unpadded elements
            output_masked = output_flat[mask_flat]
            target_masked = target_flat[mask_flat]
            
            # Calculate MSE loss on masked elements
            loss = criterion(output_masked, target_masked)
        
            val_loss += loss.item()

    return val_loss
# END VALIDATE

def pad_collate(batch):
    # Extract "data" and "nmm" from each sample in the batch
    eeg_data = [torch.tensor(sample['data'], dtype=torch.float32) for sample in batch]
    nmm_data = [torch.tensor(sample['nmm'], dtype=torch.float32) for sample in batch]
    
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

    return {'data': padded_eeg_data, 'nmm': padded_nmm_data, 'mask': attention_masks}

if __name__ == '__main__':
    main_train()


