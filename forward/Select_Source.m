% Sample 1000 sources
% Feb 2023
% Utility functions live alongside this script; add the repository root with addpath(genpath(pwd))
load('forward/test_sample_five_sources.mat');
load('anatomy/fs_cortex_20k_centre.mat')
all_dis = mydis(centre', centre');
rules = {};
%% source magnitude
rules{1} = (max(scale_ratio(:, :, 1),[],2)./min(scale_ratio(:, :, 1),[],2))'<10 & (max(scale_ratio(:, :, 2),[],2)./min(scale_ratio(:, :, 2),[],2))'<10;
%% patch differences
rules{2} = ones(1, size(selected_region, 1));
rules{3} = ones(1, size(selected_region, 1));
% for i=1:size(selected_region, 1)
%     gt_label = squeeze(selected_region(i,:,:))';
%     num_region_per_source = sum(~myisnan(gt_label),1);
%     source_pairs = nchoosek(1:size(gt_label,2),2);
%     rules{3}(i) = max(num_region_per_source) - min(num_region_per_source) < 35; % source size difference
%     for k=1:size(source_pairs,1)
%         s1 = gt_label(:,source_pairs(k,1));
%         s2 = gt_label(:,source_pairs(k,2));
%         rules{2}(i) = rules{2}(i) & min(all_dis(s1(s1<1000)+1,s2(s2<1000)+1),[],'all') > 20; % source distances
%     end
% end
%% only 5dB
rules{4} = zeros(1,size(selected_region, 1));
rules{4}(1:4:end) = 1;
%% selct index
selected_index = ones(1,size(selected_region, 1));
for i=1:4
    selected_index = selected_index & rules{i};
end
selected_index = find(selected_index);
rand_index = randperm(length(selected_index));
final_selected_index = repmat(selected_index(sort(rand_index(1:1000))), 4, 1) + [0;1;2;3];
final_selected_index = final_selected_index(:);
save("selected_index_S5.mat","final_selected_index")
%%
function y = myisnan(x)
    y = abs(x-15213)<1e-6;
end