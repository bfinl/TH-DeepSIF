clear


%% Choose the target file and leadfiled to change scale ratio

train = 1;
filename = "../source/train_998_snr15_extenE7.mat";
leadfield = '../anatomy/leadfield_64_20k.mat';
output_filename = 'train_998_snr15_extenE7_64.mat';

%% Reload all fields in the orginal file
load(filename)

save(output_filename, 'selected_region')
save(output_filename, 'nmm_idx', '-append')
save(output_filename, 'random_samples', '-append')
save(output_filename, 'current_snr', '-append')
save(output_filename, 'mag_change', '-append')
save(output_filename, 'dataset_group', '-append')

%% Scaling Factor
load(leadfield);
gt = load(filename);
scale_ratio = [];
n_source = size(gt.selected_region, 2);
parfor i=1:size(gt.selected_region, 1)
    for k=1:n_source
        a = gt.selected_region(i,k,:);
        a = a(:);
        a(a>1000) = [];
        if train
            scale_ratio(i,k,:) = find_alpha(a, [fwd_small zeros(size(fwd,1),4)], 10:2:20);
            
        else
            scale_ratio(i,k,:) = find_alpha(a, [fwd_small zeros(size(fwd,1),4)], [10,15]);
        end
    end
end

save(output_filename, 'scale_ratio', '-append')
%%
function [alpha] = find_alpha(selected_region, fwd, target_SNR)
    load('one_nmm_sample.mat')
    spike_shape = current_nmm(:,81)/max(current_nmm(:,81));
    current_nmm(:,81) = current_nmm(:,1);
    current_nmm(:, selected_region+1) = repmat(spike_shape,1,length(selected_region));
    [Ps, Pn, ~] = calcualate_SNR(current_nmm, fwd, selected_region+1, 305:405);
    alpha = sqrt(10.^(target_SNR./10).*Pn./Ps);
        
    % current_nmm(:, selected_region+1) = alpha*current_nmm(:, selected_region+1);
end




function [Ps, Pn, cur_snr] = calcualate_SNR(nmm, fwd, region_id, spike_ind)
% Caculate SNR at sensor space.
%
% INPUTS:
%     - nmm        : NMM data with single activation, time * channel
%     - fwd        : leadfield matrix, num_electrode * num_regions
%     - region_id  : source regions, start at 1, region_id(1) is the center
%     - spike_ind  : index to calculate the spike snr
% OUTPUTS:
%     - Ps         : signal power
%     - Pn         : noise power
%     - cur_snr    : current SNR in dB

    sig_eeg = (fwd(:, region_id)*nmm(:, region_id)')';   % time * channel
    sig_eeg_rm = sig_eeg - mean(sig_eeg, 1);
    dd = 1:size(nmm,2);
    dd(region_id) = [];
    noise_eeg = (fwd(:,dd)*nmm(:,dd)')';
    noise_eeg_rm = noise_eeg - mean(noise_eeg, 1);

    Ps = norm(sig_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    Pn = norm(noise_eeg_rm(spike_ind,:),'fro')^2/length(spike_ind);
    cur_snr = 10*log10(Ps/Pn);
end