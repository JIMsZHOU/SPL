function [nn,L] = spldtrain(nn, train_x, train_y, opts)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L, the sum
% squared error for each training minibatch.

assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number of input arguments must be 4 or 6')


opts.validation = 0;

if nargin == 6
    opts.validation = 1;
end

numepochs = opts.numepochs;
update = opts.update;
update2 = opts.update2; % diversity update parameter
pace2 = opts.pace2; % diversity parameter
Idx = opts.train_Idx;
cnum = opts.cnum; % cluster num

m = size(train_x, 1);
pace = 0; % SPL parameter
kk = randperm(m);

L = zeros(numepochs*7,1);
n =1;

for i = 1 : 6
    tic;
    
    if i == 1
    
        %% initalize cluster and the first training data
        batch_x = train_x(kk(1:end), :);
        batch_y = train_y(kk(1:end), :);
        batch_Idx = Idx(kk(1:end));
        
        p=zeros(cnum,1); % number of cluster
        cluster_x{cnum} = [];
        cluster_y{cnum} = [];
        for jj = 1:cnum
            for j=1:m
                if batch_Idx(j) == jj
                    p(jj,1) = p(jj,1)+1;
                    cluster_x{jj}(p(jj,1),:) = batch_x(j,:);
                    cluster_y{jj}(p(jj,1),:) = batch_y(j,:);
                end
            end        
        end

        disp(size(batch_x,1)); % display the number of easy samples selected to train
        for j =1:20
            kkk = randperm(m);
            nn = nnff(nn, batch_x(kkk(1:end),:), batch_y(kkk(1:end),:));
            nn = nnbp(nn);
            nn = nnapplygrads(nn);
        end

        %% global error
        nn = nnff(nn, batch_x, batch_y);
        for j = 1:1:m
            glbe(j,1) = sum(abs(nn.e(j))); %#ok<AGROW>
        end
        [glbeas,~] = sort(glbe);
        pace = median(glbeas);
    end
    
    clear batch_x;
    clear batch_y;
    clear pp;

    pp = 0;
    k{cnum}=[];
    glbe_o{cnum}=[];

    if i ~= 6
        for jj = 1:cnum
            if isempty(cluster_x{jj})
                continue;
            end
            nn = nnff(nn, cluster_x{jj}, cluster_y{jj});
            for j =1:1:p(jj,1)
                glbe_o{jj}(j,1) = sum(abs(nn.e(j)));
            end
            [glbeas, k{jj}] = sort(glbe_o{jj}); %#ok<ASGLU>
            for j =1:1:p(jj,1)
                if glbe_o{jj}(k{jj}(j)) <= pace +pace2/(sqrt(j)+sqrt(j-1))
                    pp = pp +1;
                    batch_x(pp,:) = cluster_x{jj}(k{jj}(j),:);
                    batch_y(pp,:) = cluster_y{jj}(k{jj}(j),:);
                end
            end
        end
    end

    if i == 6
        numepochs = 2*numepochs;
        batch_x = train_x;
        batch_y = train_y;
    end

    finaln = size(batch_x,1);
    disp(finaln); % display the number of easy samples selected to train
    
    for j = 1:numepochs
        finalord=randperm(finaln);
        batch_x = batch_x(finalord(1:end),:);
        batch_y = batch_y(finalord(1:end),:);
        nn = nnff(nn, batch_x, batch_y);
        nn = nnbp(nn);
        nn = nnapplygrads(nn);

        L(n) = nn.L;
        n = n+1;
        
    end
    
    t = toc;
        
    disp(['epoch ' num2str(i) '/' num2str(6) '. Took ' num2str(t) ' seconds' ]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
    
    % update the pace value
    pace = update*pace;
    pace2 = update2*pace2;
end
end

