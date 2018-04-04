function contract_SDD

	load SDD SDD_input SDD_target;

	in_x= SDD_input;
	in_y= SDD_target;

	inlayer = size(in_x', 1); % feature num
	outlayer = size(in_y', 1); % target num

	num = size(in_x, 1); % whole num of data
	cnum= floor(num/1000); % SPLD cluster number
    cnum2 =floor(cnum/outlayer); % NSPLD cluster number

	nn1 = nnsetup([inlayer 100 outlayer]); %originBP
	nn2 = nnsetup([inlayer 100 outlayer]); %SPLBP
	nn3 = nnsetup([inlayer 100 outlayer]); %SPLD
	nn4 = nnsetup([inlayer 100 outlayer]); %NSPLD

	p = zeros(outlayer, 1); % num of each class
	middle_x{outlayer}=[];
	middle_y{outlayer}=[];
	for ii = 1:outlayer
		for i = 1:num
			[~, expected] = max(in_y(i,:),[],2);
			if expected == ii
				p(ii,1) = p(ii,1)+1;
				middle_x{ii}(p(ii,1),:) = in_x(i,:);
				middle_y{ii}(p(ii,1),:) = in_y(i,:);
			end
		end	
	end

	Idx_other{outlayer} = 0;
	for i = 1:outlayer
		Idx_other{i}= kmeans(middle_x{i}, cnum2);
	end
	
	for i = 1:outlayer
        if i == 1
            train_x1 = middle_x{i};
            train_y1 = middle_y{i};
            train_Idx1 = Idx_other{i};
        end
        if i ~= 1
            train_x1 = [train_x1;middle_x{i}]; %#ok<AGROW>
            train_y1 = [train_y1;middle_y{i}]; %#ok<AGROW>
            train_Idx1 = [train_Idx1; Idx_other{i}]; %#ok<AGROW>
        end
	end
	
	k = randperm(num);
	train_x = train_x1(k(1:50000),:);
	train_y = train_y1(k(1:50000),:);
	test_x = train_x1(k(50001:end),:);
	test_y = train_y1(k(50001:end),:);

	Idx = kmeans(in_x, cnum); % pre cluster the training data
	train_Idx = Idx(k(1:50000),:); % index of training data when random.
	train_Idx1 = train_Idx1(k(1:50000),:);

	% [train_x, ~] = mapminmax(train_x',0,1);
	[train_x, mu, sigma] = zscore(train_x);
	test_x = normalize(test_x, mu, sigma);

	%% originBP
	opts1.numepochs = 1000;
	opts1.batchsize = 50000;
	[nn1, L1] = nntrain(nn1, train_x, train_y, opts1);
	[er1, ~] = nntest(nn1, test_x, test_y);

	%% SPLBP
	opts2.update = 1.8;
	opts2.numepochs = 400;
	[nn2, L2] = spltrain(nn2, train_x, train_y, opts2);
	[er2, ~] = nntest(nn2, test_x, test_y);

	%% SPLD
	opts3.numepochs = 400;
	opts3.update = 1.8;
	opts3.update2 = 1.6;
	opts3.pace2 = 0.001;
	opts3.train_Idx = train_Idx;
	opts3.cnum = cnum;
	[nn3, L3] = spldtrain(nn3, train_x, train_y, opts3);
	[er3, ~] = nntest(nn3, test_x, test_y);

	%% NSPLD
	opts4.numepochs = 400;
	opts4.update = 1.8;
	opts4.train_Idx = train_Idx1;
    opts4.cnum = cnum2;
	[nn4, L4] = nspldtrain(nn4, train_x, train_y, opts4);
	[er4, ~] = nntest(nn4, test_x, test_y);

    L1 =L1';
    L2 =L2';
    L3 =L3';
    L4 =L4';

    LA = [L2;L3;L4];
    save L1 L1;
    save LA LA;

	disp(1-er1);
    disp(1-er2);
    disp(1-er3);
    disp(1-er4);
end