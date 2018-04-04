function contrast

	load MRI_AD1 fea gnd;

    cnum=5;
	num = size(fea, 1);

	gnd3d = zeros(num,3);
	for i = 1:1:num
	    switch gnd(i,:)
	        case 0
	            gnd3d(i,:) = [1 0 0];
	        case 1
	            gnd3d(i,:) = [0 1 0];
	        case 2
	            gnd3d(i,:) = [0 0 1];
	    end
	end

	inlayer = size(fea', 1);
	outlayer = size(gnd3d', 1);

	nn1 = nnsetup([inlayer 100 outlayer]); %originBP
	nn2 = nnsetup([inlayer 100 outlayer]); %SPLBP
	nn3 = nnsetup([inlayer 100 outlayer]); %SPLD

	Idx = kmeans(fea, cnum); % pre cluster the training data
	k = randperm(num);
	train_x = fea(k(1:500),:);
	train_y = gnd3d(k(1:500),:);
	test_x = fea(k(501:end),:);
	test_y = gnd3d(k(501:end),:);
	train_Idx = Idx(k(1:500),:); % index of training data when random.

	% [train_x, ~] = mapminmax(train_x',0,1);
	[train_x, mu, sigma] = zscore(train_x);
	test_x = normalize(test_x, mu, sigma);

	%% originBP
	opts1.numepochs = 400;
	opts1.batchsize = 500;
	[nn1, ~] = nntrain(nn1, train_x, train_y, opts1);
	[er1, bad2] = nntest(nn1, test_x, test_y);

	%% SPLBP
	opts2.update = 1.02;
	opts2.numepochs = 400;
	nn2 = spltrain(nn2, train_x, train_y, opts2);
	[er2, bad2] = nntest(nn2, test_x, test_y);

	%% SPLD

	opts3.numepochs = 400;
	opts3.update = 1.04;
	opts3.update2 = 1.01;
	opts3.pace2 = 0.01;
	opts3.train_Idx = train_Idx;

	nn3 = spldtrain(nn3, train_x, train_y, opts3);
	[er3, bad3] = nntest(nn3, test_x, test_y);

	disp(1-er1);
    disp(1-er2);
    disp(1-er3);
end