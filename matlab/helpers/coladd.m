function A = coladd(A, c)
% COLADD Adds the column vector C to each column in A
%   X = COLADD(A,C). A is an m x n matrix and C is a m x 1 vector. X(:,i) =
%   A(:,i) + c for i = 1, ..., n

[Ar,Ac] = size(A);

if(size(c,2) ~= 1)
    error('c must be a column vector');
elseif(Ar ~= size(c,1))
    error('A and c must have the same number of rows');
end
% 
% if(Ar <= Ac) % Fewer rows than cols ==> loop over rows
%     for(i = 1:Ar)
%         A(i,:) = A(i,:) + c(i);
%     end
% else
%     for(j = 1:Ac)
%         A(:,j) = A(:,j) + c;
%     end
% end
%     

A = A + repmat(c,1,Ac);