function MySPL

%% load data set
load mnist_uint8;

inlayer = size(train_x', 1);
outlayer = size(train_y', 1);
num = size(train_x, 1);

k = randperm(num);

train_x = double(train_x(1:end,:));
test_x  = double(test_x(1:end,:));
train_y = double(train_y(1:end,:));
test_y  = double(test_y(1:end,:));

%% normalize and make data in [0,1]
[train_x, train_xps] = mapminmax(train_x',0,1);
[train_x, mu, sigma] = zscore(train_x');
test_x = normalize(test_x, mu, sigma);

%% train 2 times in order to initilize reasonable weight
% rand('state',0);
nn = nnsetup([inlayer 40 outlayer]);
% nn.output = 'softmax';
opts.numepochs = 5;
opts.batchsize = 60000;
opts.plot = 1;
[nn, ~] = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

%% SPL train 
% rand('state',0);
opts.numepochs = 100;
opts.update = 1.15;
nn = spltrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

% assert(er < 0.1, 'Too big error');



end


