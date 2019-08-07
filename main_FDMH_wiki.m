clear;clear memory;
addpath('./tools')
nbits_set = [16];%[8,16,32,48,64,96,128];

%% Load dataset
load('wikiData.mat')
X{1} = [I_tr;I_te];
X{2} = [T_tr;T_te];
gnd = [L_tr;L_te];

view_num = size(X,2);
n_anchor = 500;
Anchor = cell(1,view_num);
n_Sam = size(X{1},1);
for it = 1:view_num
    X{it} = double(X{it});
    anchor = X{it}(randsample(n_Sam, n_anchor),:);
    Dis = EuDist2(X{it},anchor,0);
    sigma = mean(mean(Dis)).^0.5;
    feavec = exp(-Dis/(2*sigma*sigma));
    X{it} = bsxfun(@minus, feavec', mean(feavec',2));
end

% Separate Train and Test Index
tt_num = size(I_te,1);
data_our.gnd = gnd;
tt_idx = n_Sam-tt_num+1:n_Sam;
tr_idx = 1:n_Sam-tt_num; 
ttgnd = gnd(tt_idx,:);
trgnd = gnd(tr_idx,:);
clear gnd;

for ii=1:length(nbits_set)
    nbits = nbits_set(ii);
    data_our.indexTrain= tr_idx;
    data_our.indexTest= tt_idx;
    ttfea = cell(1,view_num);
    for view = 1:view_num
        data_our.X{view} = normEqualVariance(X{view}')';
        ttfea{view} = data_our.X{view}(:,tt_idx);
    end

    for n_iters = 1:5
        pars.beta     = .5; 
        pars.gamma    = 100; 
        pars.lambda = .01;
        pars.Iter_num = 5;
        pars.nbits    = nbits;
        pars.r = 3;
        pars.sigma = 5;
        
        [B_trn, W, U_W, R, P, alpha, trtime] = mvFDMH_fun(data_our,pars);
        
        % for testing
        V = zeros(nbits,tt_num);
        for ind = 1:size(ttfea,2)
            V = V+alpha(ind)*W{ind}*ttfea{ind};
        end
        B_tst = V'*U_W >0;
        
        % Groungdtruth
        WtrueTestTraining = bsxfun(@eq, ttgnd, trgnd');
        
        %% Evaluation
        B1 = compactbit(B_trn);
        B2 = compactbit(B_tst);
        DHamm = hammingDist(B2, B1);
        [~, orderH] = sort(DHamm, 2);
        MAP = calcMAP(orderH, WtrueTestTraining);
        fprintf('iter = %d, Bits: %d, MAP: %.4f...   \n', n_iters, nbits, MAP);

    end
end