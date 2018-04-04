function NSPLD
	%% load data
    load MRI_AD1 fea gnd;
    cnum =3;
    num = size(fea, 1);
    gnd3d = zeros(num,3);

    %%set 3 dimensions and classfication
    p1=0;
    p2=0;
    p3=0;
    for i = 1:1:num
        switch gnd(i,:)
            case 0
            	p1 = p1+1;
                gnd3d(i,:) = [1 0 0];
                fea1(p1,:) = fea(i,:);
                gnd3d1(p1,:) = gnd3d(i,:);
            case 1
                gnd3d(i,:) = [0 1 0];
                fea2(p2,:) = fea(i,:);
                gnd3d2(p2,:) = gnd3d(i,:);
            case 2
                gnd3d(i,:) = [0 0 1];
                fea3(p3,:) = fea(i,:);
                gnd3d3(p3,:) = gnd3d(i,:)；
        end
    end

    inlayer = size(fea', 1);
    outlayer = size(gnd3d', 1);
    nn = nnsetup([inlayer 100 outlayer]);

    Idx1 = kmeans(fea1, cnum); % pre cluster the training data 1
    Idx2 = kmeans(fea2, cnum); % pre cluster the training data 2
    Idx3 = kmeans(fea3, cnum); % pre cluster the training data 3

    %% begining of training
    k = randperm(num);
    train_x = fea(k(1:500),:); % training data
    train_y = gnd3d(k(1:500),:);
    test_x = fea(k(501:end),:);
    test_y = gnd3d(k(501:end),:);
    train_Idx = Idx(k(1:500),:); % index of training data when random.

    [train_x, mu, sigma] = zscore(train_x);
    test_x = normalize(test_x, mu, sigma);
        
    %% SPLD

    opts.numepochs = 400;
    opts.update = 1.02;
    opts.update2 = 1.04;
    opts.pace2 = 0.01;
    opts.train_Idx = train_Idx;

    nn = spldtrain(nn, train_x, train_y, opts);
    [er, bad] = nntest(nn, test_x, test_y);







end