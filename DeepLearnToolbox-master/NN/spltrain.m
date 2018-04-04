function [nn,L] = spltrain(nn, train_x, train_y, opts)
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
m = size(train_x, 1);
pace = 0;
kk = randperm(m);
L = zeros(numepochs*7,1);
n =1;

for i = 1 : 6 % six time train every samples
    tic;
    
    % 1st time calculate pace and train whole dataset 5 times initalize the weight
    if i ==1
        batch_x = train_x(kk(1:end),:);
        batch_y = train_y(kk(1:end),:);
        
        disp(size(batch_x,1)); % display the number of easy samples selected to train
        for j =1:20
            kkk = randperm(m);
            nn = nnff(nn, batch_x(kkk(1:end),:), batch_y(kkk(1:end),:));
            nn = nnbp(nn);
            nn = nnapplygrads(nn);
        end
        
        nn = nnff(nn, batch_x, batch_y);
        for j = 1:1:m
            glbe(j,1) = sum(abs(nn.e(j))); %#ok<AGROW>
        end
        [glbeas,~] = sort(glbe);
        pace = median(glbeas);
    end

    % calculate the data which the vweight equals 1

    clear p;
    clear batch_x;
    clear batch_y;

    p = 0;
    if i ~= 6
        nn = nnff(nn, train_x, train_y);
        for j = 1:m
            if sum(abs(nn.e(j))) < pace
                p = p+1;
                batch_x(p,:) = train_x(j,:);
                batch_y(p,:) = train_y(j,:);
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
end
end

