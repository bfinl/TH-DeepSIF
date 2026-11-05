function nn = mynorm(x, varargin)
% Calcualte norm over the rows/columns of a matrix
% x: defalut, d*n, calculated the norm over each column
% the second input is the dimension
if nargin > 1
    nn = sqrt(sum(x.^2,varargin{1}));
else
    nn = sqrt(sum(x.^2, 1));
end

end