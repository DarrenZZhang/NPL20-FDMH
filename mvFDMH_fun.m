function [B_train, W, U_W, R, P, alpha, trtime] = mvFDMH_fun(data, pars)

gamma     = pars.gamma; 
beta     = pars.beta; 
lambda   = pars.lambda;
Iter_num = pars.Iter_num;
nbits    = pars.nbits;
sigma    = pars.sigma;
trIdx = data.indexTrain;
r = pars.r;

view_num = size(data.X,2);
XXT = cell(1,view_num);
dim = zeros(1,view_num);
our_data = cell(1,view_num);
for ind = 1:view_num
    our_data{ind} = data.X{ind}(:,trIdx);
    [dim(ind), n] = size(our_data{ind});
    XXT{ind} = our_data{ind}*our_data{ind}';
end

% label matrix Y = c x N   (c is amount of classes, N is amount of instances)
if isvector(data.gnd)
    L_tr = data.gnd(trIdx);
    Y = sparse(1:length(L_tr), double(L_tr), 1); Y = full(Y');
else
    L_tr = data.gnd(trIdx,:);
    Y = double(L_tr');
end

% %%%%%% B - initialize %%%%%%
B = randn(nbits, n)>0; B = B*2-1;
P = randn(nbits, size(Y,1));  %G=Y'*B;
V = P*Y;
W = cell(1,view_num);
alpha = ones(view_num,1)/view_num;

tic;
u = ones(nbits,1);
z = ones(nbits,1);
%------------------------training----------------------------
for iter = 1:Iter_num
    fprintf('The %d-th iteration...\n',iter);
    B0 = B; Z = diag(z);
    alpha_r = alpha.^r;
    
    % ----------------------- W-step -----------------------%
    WX = zeros(nbits,n);
    for ind = 1:size(our_data,2)
        W{ind} = V*our_data{ind}'/(XXT{ind}+lambda*eye(dim(ind)));
        WX = WX+alpha_r(ind)*W{ind}*our_data{ind};
    end
    
    % ----------------------- R-step -----------------------%
    
    [U0,~,P0] = svd(V*B', 'econ');
    R = U0*P0';
    
    % ----------------------- P-step -----------------------%
     [U1,~,P1] = svd(Z*B*Y', 'econ');
     P = U1*P1';
%    P = B*Y'/(YYT+pars.theta*eye(c));
    
    % ----------------------- V-step -----------------------%
    V = (sum(alpha_r)*eye(nbits)+beta*R'*R)\(WX + beta*R'*B);
    
    % ----------------------- B-step -----------------------%
    A = beta*R*V+gamma*Z*P*Y; mu = median(A,2); A = bsxfun(@minus, A, mu);
    B = sign(A);
    
    % ----------------------- alpha-step -----------------------%
    h = zeros(view_num,1);
    for view = 1:view_num
        h(view) = norm(V - W{view}*our_data{view},'fro')^2;
    end
    H = bsxfun(@power,h,1/(1-r));
    alpha = bsxfun(@rdivide, H, sum(H));
    
    % ----------------------- z-step -----------------------%
    z21 = sum((B-P*Y).^2,2);
    z = exp(-z21/(sigma^2));

%     f(iter) = norm(B-B0);
end
trtime = toc;
B_train = B'>0;

% Out-of-Sample
V0 = zeros(nbits,n);
for ind = 1:view_num
    V0 = V0+alpha_r(ind)*W{ind}*our_data{ind};
end
NT = (V0*V0' + 1 * eye(size(V0,1))) \ V0;
U_W = NT*B_train;

end