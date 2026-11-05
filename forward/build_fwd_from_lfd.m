
lfd_free = bs_headmodel.Gain(1:306,:); % if you are using brainstorm
% lfd_free can also be calculated from curry
lfd = lfd_to_fwd(lfd_free,bs_headmodel.GridOrient);
ori = bs_headmodel.GridOrient;
fwd = fwd_to_rmfwd(lfd, rm);



function fwd = lfd_to_fwd(lfd, ori, tri, curryori)
% Transfer the free direction leadfield to fix direction

% INPUTS:
%     - lfd        : free direction leadfield, num_electrode * (num_vertice*3)
% OUTPUTS:
%     - v          : direction between two points; size 1*3


% Inputs:
% lfd: num_senser * (num_vertice*3)
% ori: num_vertice * 3, for curry's case, ori could have more points than
% num_vertice, but it would be at the end, so we can just ignore it.
% Or input the pos, tri pair as the second and third parameter and ori will
% be calculated from these.
% 01222020 Update the input format to include the choice of input pos and
% tri

if nargin == 3
    % assume the input is the position and the triangulation
    pos = ori;
    [ori] = ComputeVertexNorm(pos, tri);
    if sum(isnan(ori(:)))
        disp('NaN value in ori, quit')
        fwd=0;
        return
    end
end 

if nargin == 4
    % assume the input is the position and the triangulation, the last is
    % of size (num_vertices*3)
    pos = ori;
    [ori] = ComputeVertexNorm(pos, tri);
    if sum(isnan(ori(:)))
        disp('NaN value in ori, quit')
        fwd=0;
        return
    end
    ori = reshape(sum(reshape(repmat(ori,1,3)',3,[])'.*curryori,2),3,[])';
end 
    
num_vertices = size(lfd,2)/3;
fwd = zeros(size(lfd,1), num_vertices);
for i = 1:num_vertices
    fwd(:,i) = lfd(:, (i-1)*3+1:i*3) * ori(i,:)';
end
end


function new_fwd = fwd_to_rmfwd(fwd, rm)
new_fwd = zeros(size(fwd,1), length(unique(rm)));
for i=0:993
    new_fwd(:,i+1) = sum(fwd(:, rm==i),2);
end
end