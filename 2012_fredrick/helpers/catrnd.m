function X = catrnd(p, varargin)
% CATRND    Categorical random number generator
%  X = CATRND(P) generates a random number from the categorical
%  distribution with probabilities {P(i)}_{i=1}^K and index set {1, ..., K}
%
%  X = CATRND(P,N) generates N realisations from the categorical
%  distribution as described above.

if(nargin > 1)
    N = varargin{1};
else
    N = 1;
end

bins = cumsum(p);
[~, X] = histc(rand(N,1), [0 ; bins(:)]);
