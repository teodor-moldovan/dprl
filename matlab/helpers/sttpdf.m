function q = sttpdf(x, nu, Lambda)
% STTPDF  Computes the PDF for a Student's t-distribution
%   Q = STTPDF(X, NU, LAMBDA) takes a matrix (nx x N) of N column vectors and computes the value of the
%   multivariate (nx-dim) Student's t-PDF with precision LAMBDA, NU degrees of freedom and zero mean.
% 
%   Fredrik Lindsten, 2011-04-11
%   lindsten@isy.liu.se

nx = size(Lambda,1);
D2 = sum(x.*(Lambda*x),1);
q = (1 + D2/nu).^(-1/2*(nu + nx));
normConst = gammaln(nu/2 + nx/2) - gammaln(nu/2) + 1/2*log(det(Lambda)) - nx/2*(log(pi) + log(nu));
q = exp(normConst).*q;