function [nn,L] = nspldtrain(nn, train_x, train_y, opts)
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
update = opts.update;  % SPL update parameter
Idx = opts.train_Idx;
cnum = opts.cnum; % num of cluster for each class


m = size(train_x, 1);
pace = 0; % SPL parameter
kk = randperm(m);
outlayer = nn.size(1,nn.n);

L = zeros(numepochs*7,1);
n =1;


for i = 1 : 6
    tic;
    
    if i == 1

    
        %% initalize cluster and the first training data
        batch_x = train_x(kk(1:end), :);
        batch_y = train_y(kk(1:end), :);
        batch_Idx = Idx(kk(1:end));

        p = zeros(outlayer,1);
        class_x{outlayer} =[];
        class_y{outlayer} =[];
        N_id{outlayer}=[];
        for jj = 1:outlayer % the origin class of dataset
            for j = 1:m % num of dataset
                [dummy, expected] = max(batch_y(j,:),[],2);
                if expected == jj
                    p(jj,1) = p(jj,1)+1;
                    class_x{jj}(p(jj,1),:) = batch_x(j,:);
                    class_y{jj}(p(jj,1),:) = batch_y(j,:);
                    N_id{jj}(p(jj,1),:) = batch_Idx(j,:);
                end
            end
        end

        num = zeros(outlayer,cnum);
        pace = zeros(outlayer,cnum);
        cluster_x{outlayer,cnum} = [];
        cluster_y{outlayer,cnum} = [];
        glbe_o{outlayer,cnum} = [];

        for jjj = 1:outlayer % the origin class of dataset
            for jj = 1:cnum % the num of Kmeans class
                for j = 1:p(jjj,1)
                    if N_id{jjj}(j,1) == jj
                        num(jjj,jj) = num(jjj,jj)+1;
                        cluster_x{jjj,jj}(num(jjj,jj),:) = class_x{jjj}(j,:);
                        cluster_y{jjj,jj}(num(jjj,jj),:) = class_y{jjj}(j,:);
                    end
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
        
        for jjj = 1:outlayer
            for jj = 1:cnum
                if isempty(cluster_x{jjj,jj})
                    continue;
                end
                nn = nnff(nn, cluster_x{jjj,jj}, cluster_y{jjj,jj});
                for j =1:num(jjj,jj)
                    glbe_o{jjj,jj}(j,1) = sum(abs(nn.e(j)));
                end
                [glbeas,~] = sort(glbe_o{jjj,jj});
                pace(jjj,jj) = median(glbeas);
            end
        end
    end
    
    clear batch_x;
    clear batch_y;
    clear p;
    p = 0;

    if i ~= 6
        for jjj = 1:outlayer
            for jj = 1:cnum
                if isempty(cluster_x{jjj,jj})
                    continue;
                end
                nn = nnff(nn, cluster_x{jjj,jj}, cluster_y{jjj,jj});
                for j = 1:num(jjj,jj)
                    if sum(abs(nn.e(j))) <= pace(jjj,jj)
                        p = p+1;
                        batch_x(p,:) = cluster_x{jjj,jj}(j,:);
                        batch_y(p,:) = cluster_y{jjj,jj}(j,:);
                    end
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
    
    % update the pace value
    pace = update*pace;

    disp(['epoch ' num2str(i) '/' num2str(6) '. Took ' num2str(t) ' seconds' ]);
    nn.learningRate = nn.learningRate * nn.scaling_learningRate; 
end
end

