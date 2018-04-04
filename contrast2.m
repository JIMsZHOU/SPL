function contrast2

	load SDD SDD_input SDD_target;

    cnum=5;
	num = size(SDD_input, 1);

% 	gnd3d = zeros(num,3);
% 	for i = 1:1:num
% 	    switch gnd(i,:)
% 	        case 0
% 	            gnd3d(i,:) = [1 0 0];
% 	        case 1
% 	            gnd3d(i,:) = [0 1 0];
% 	        case 2
% 	            gnd3d(i,:) = [0 0 1];
% 	    end
% 	end

	inlayer = size(SDD_input', 1);
	outlayer = size(SDD_target', 1);
	nn1 = nnsetup([inlayer 100 outlayer]); %originBP
	nn2 = nnsetup([inlayer 100 outlayer]); %SPLBP
	nn3 = nnsetup([inlayer 100 outlayer]); %SPLD

	Idx = kmeans(SDD_input, cnum); % pre cluster the training data
	k = randperm(num);
	train_x = SDD_input(k(1:50000),:);
	train_y = SDD_target(k(1:50000),:);
	test_x = SDD_input(k(50001:end),:);
	test_y = SDD_target(k(50001:end),:);
	train_Idx = Idx(k(1:50000),:); % index of training data when random.

	% [train_x, ~] = mapminmax(train_x',0,1);
	[train_x, mu, sigma] = zscore(train_x);
	test_x = normalize(test_x, mu, sigma);

	%% originBP
	opts1.numepochs = 400;
	opts1.batchsize = 50000;
    rand('state',0);
	[nn1, ~] = nntrain(nn1, train_x, train_y, opts1);
	[er1, bad2] = nntest(nn1, test_x, test_y);

	%% SPLBP
	opts2.update = 1.04;
	opts2.numepochs = 400;
    rand('state',0);
	nn2 = spltrain(nn2, train_x, train_y, opts2);
	[er2, bad2] = nntest(nn2, test_x, test_y);

	%% SPLD

	opts3.numepochs = 400;
	opts3.update = 1.04;
	opts3.update2 = 1.02;
	opts3.pace2 = 0.01;
	opts3.train_Idx = train_Idx;
    rand('state',0);
	nn3 = spldtrain(nn3, train_x, train_y, opts3);
	[er3, bad3] = nntest(nn3, test_x, test_y);

	disp(1-er1);
    disp(1-er2);
    disp(1-er3);
end