function dis = mydis(x,y,varargin)
% x: 3*n
% y: 3*m
% y is smaller is better
% dis: distance between each point in x with y  n*m matrix
if nargin ==2 
    dis = ones(size(x,2), size(y,2));
    for i = 1:size(y,2)
        dis(:,i) = sqrt(sum((x - repmat(y(:,i),1,size(x,2))).^2))';
    end
else
    dis = ones(size(x,2), 1);
    for i = 1:size(y,2)
        dis(i) = sqrt(sum((x(:,i) - y(:,i)).^2));
    end
end