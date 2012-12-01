function [z_log, numComp, prm, prs2] = segment_nonlin_dplr(data,prior,par)
%% DP linear regression model - joint clustering on (X,Y)
%  Run a standard Gibbs sampler based on CRP
z_log = zeros(par.T, par.numMCMC);
numComp = zeros(1, par.numMCMC);
Y = [data.x ; data.u ; data.y]; % We view the data as jointly Gaussian

%% Initializew
K = min(100,par.T);
zmax = K;
active_z = 1:K;
z = kmeans(Y',K)';

%%
% Sample the parameters  th | c, Y
[mu, Sigma] = gibbs_th_sampler(Y, z, K, prior);

%% CRP-settings
for(r = 2:par.numMCMC)    
    % Sample new labels      z_t | z_{-t}, th, Y
    [z, active_z, zmax] = gibbs_z_sampler(Y, z, mu, Sigma, prior, active_z, zmax);
    K = length(active_z);
    numComp(r) = K;
    z_log(:,r) = z(:);
    
    % Sample the parameters  th | c, Y
    [mu, Sigma] = gibbs_th_sampler(Y, z, K, prior);    
    
    %=======
    if(par.plotOn)
        indx = 3; % \theta
        indy = 4; % \ddot\theta
%         indy = 2; % \ddot x
        ind = [indx indy+5];
        mu1 = mu(ind,:,:);
        Sigma1 = Sigma(ind,ind,:);
        figure(2);
        plot(data.x(indx,:),data.y(indy,:),'.','color',0.7*[1 1 1]); hold on;
        for(k = 1:K)
            plot(mu1(1,:,k), mu1(2,:,k),'k','marker','x','markersize',14,'linewidth',2);
            H = error_ellipse(Sigma1(:,:,k), mu1(:,:,k));
            set(H,'color','b','linestyle','--','linewidth',1);
        end
        hold off;
        
                
        title(sprintf('iter = %i, #comp = %i, k* = %i', r, numComp(r),K));
        hold off;
        drawnow;
    end
end
end
%--------------------------------------------------------------------------
function [mu, Sigma] = gibbs_th_sampler(Y, z, K, prior)
% Sample from the parameter posterior for the DPMM model - NIW prior
d = size(Y,1);
dy = 1:d;
dx = d+1;

mu = zeros(d,1,K);
Sigma = zeros(d,d,K);

for(k = 1:K)
    ik = (z == k);
    nk = sum(ik);
    
    nuk = prior.nu0 + nk;
    Dk = [Y(:,ik) ; ones(1,nk)];
    Vk  = prior.V0 + Dk*Dk';
    Sk  = Vk(dy,dy) - Vk(dy,dx)*(Vk(dx,dx)\Vk(dx,dy));

    Sigma(:,:,k) = iwishrnd((Sk+Sk')/2, nuk);
    mu(:,:,k) = mvnrnd(Vk(dy,dx)/Vk(dx,dx), Sigma(:,:,k)/Vk(dx,dx));
end

end
% -------------------------------------------------------------------------
function [z, active_z, zmax] = gibbs_z_sampler(Y, z, mu, Sigma, prior, active_z, zmax)

K = length(active_z);
[d,T] = size(Y);
dy = 1:d;
dx = d+1;

q = zeros(T, K);
% Compute p(Y_i | th_k) for t = 1...T (rows) and k = 1...K (cols)
for(k = 1:K)
    q(:,k) = mvnpdf(Y', mu(:,:,k)', Sigma(:,:,k));
end

% Some parameters of the posterior/marginal
nu1 = prior.nu0 + 1;
py = sttpdf(coladd(Y,-prior.m0), prior.nu0-d+1, (prior.nu0-d+1)*prior.kappa0/(1+prior.kappa0)*inv(prior.S0));

for(t = 1:T)
    Miz = histc(z([1:t-1 t+1:T]), 1:K); % Cluster counts, z(t) excluded
    
    if(Miz(z(t)) == 0) % No other observations are associated with label z_t - remove it from the list of active ones
        ind = [1:z(t)-1 z(t)+1:K];
        q = q(:, ind);
        Miz = Miz(ind);
        active_z = active_z(ind);
        z(z > z(t)) = z(z > z(t)) -1; % Decrease the leading components numbers by 1
        K = K - 1;
    end
        
    qnow = q(t,:).*Miz;
    qnew = prior.alpha*py(t);
    qnow = [qnow qnew];
    qnow = qnow/sum(qnow);
    % Sample a new value for z
    z(t) = catrnd(qnow);
    
    if(z(t) > K) % We spawn a new component        
        K = K+1;
        zmax = zmax + 1;
        active_z = [active_z zmax];
        
        % Posterior for the new cluster
        V1 = prior.V0 + [Y(:,t) ; 1]*[Y(:,t) ; 1]';
        S1  = V1(dy,dy) - V1(dy,dx)*(V1(dx,dx)\V1(dx,dy));
        Sigma = cat(3, Sigma, iwishrnd(S1, nu1));
        mu = cat(3, mu, mvnrnd(V1(dy,dx)/V1(dx,dx), Sigma(:,:,K)/V1(dx,dx))');
        q = [q mvnpdf(Y', mu(:,:,K)', Sigma(:,:,K))];
    end
end
end



