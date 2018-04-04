function comparision
load('MRI_Baseline.mat');
load('LabelData.mat');
r=640;
k=0;
m=0;
t=size(MRI_Baseline,2);
for i=1: r
    temp(i,1)=find(RID==RID_Baseline(i,1));
end
gnt=label_matrix(temp(1:end),:);
gnd=gnt(:,1);
my_batch=64;
my_expoch=200;
idx=randperm(r);
to_x=MRI_Baseline(idx(1:end),:);
to_y=gnd(idx(1:end),:);
my_max=max(gnd(:,1)); 
opts.numepochs = my_expoch;  
opts.batchsize=my_batch;
er_1=0;
er_2=0;
er_3=0;
nn = nnsetup([t 100 my_max]);
nn1=nn;
nn2=nn;
nn3=nn;
cvo=cvpartition(to_y,'KFold',10);
err=zeros(cvo.NumTestSets,1);
for i=1:cvo.NumTestSets
    train_index=cvo.training(i);
    test_index=cvo.test(i);
    % the data use to train the first way 
    train_x_1=MRI_Baseline(train_index(1:end),:);
    train_y_1=gnd(train_index(1:end),:);
    test_x_1=MRI_Baseline(test_index(1:end),:);
    test_y_1=gnd(test_index(1:end),:);
    % the data uses to train the second way
    train_x_2=MRI_Baseline(train_index(1:end),:);
    train_y_2=gnd(train_index(1:end),:);
    test_x_2=MRI_Baseline(test_index(1:end),:);
    test_y_2=gnd(test_index(1:end),:);
    % the data uses to train the third way
    train_x_3=MRI_Baseline(train_index(1:end),:);
    train_y_3=gnd(train_index(1:end),:);
    test_x_3=MRI_Baseline(test_index(1:end),:);
    test_y_3=gnd(test_index(1:end),:);
    % calculate the first data 
    r1=size(train_y_1,1);
    p1=size(test_y_1,1);
    train_y_1=[zeros(r1,my_max-1),train_y_1];
    for i=1:r1
        h=train_y_1(i,my_max);
        if (h<2)
        train_y_1(i,my_max)=0; 
        end
        train_y_1(i,h)=1;
        if(h<my_max)
        train_y_1(i,my_max)=0;
        end
    end
    test_y_1=[zeros(p1,my_max-1),test_y_1];
    for i=1:p1
        h=test_y_1(i,my_max);
        if (h<2)
        test_y_1(i,my_max)=0; 
        end
        test_y_1(i,h)=1;
        if(h<my_max)
        test_y_1(i,my_max)=0;
        end
    end
    [train_x_1, mu, sigma] = zscore(train_x_1);
    test_x_1 = normalize(test_x_1, mu, sigma);
    % calculate the second data
    r2=size(train_y_2,1);
    p2=size(test_y_2,1);
    train_y_2=[zeros(r1,my_max-1),train_y_2];
    for i=1:r2
        h=train_y_2(i,my_max);
        if (h<2)
        train_y_2(i,my_max)=0; 
        end
        train_y_2(i,h)=1;
        if(h<my_max)
        train_y_2(i,my_max)=0;
        end
    end
    test_y_2=[zeros(p2,my_max-1),test_y_2];
    for i=1:p2
        h=test_y_2(i,my_max);
        if (h<2)
        test_y_2(i,my_max)=0; 
        end
        test_y_2(i,h)=1;
        if(h<my_max)
        test_y_2(i,my_max)=0;
        end
    end
    [train_x_2, mu, sigma] = zscore(train_x_2);
    test_x_2 = normalize(test_x_2, mu, sigma);
    %the first way
    [nn1, L1] = nntrain(nn1, train_x_1, train_y_1, opts);  
    [er1, bad1] = nntest(nn1, test_x_1, test_y_1);
    er_1=er_1+er1;
    % the second way 
    [nn2,L2] = sstrain(nn2, train_x_2, train_y_2, opts);
    s2=sum(nn2.e.*nn2.e,2);
    [c2,index]=sort(s2);
    lanbada2=median(c2);
    temp2=find(c2<lanbada2);
    o2=size(temp2,1);
    temp_x=train_x_2(index(1:o2),:);
    temp_y=train_y_2(index(1:o2),:);
    h2=lanbada2;
    j=size(train_x_2,1);
    for i=1:100
    [nn2, L2] = sstrain(nn2, temp_x, temp_y, opts);
    s2=sum(nn2.e.*nn2.e,2); 
    h2=h2*1.8;
    temp=find(c2<h2);
    o2=size(temp,1)
    temp_x=train_x_2(index(1:o2),:);
    temp_y=train_y_2(index(1:o2),:);
    if (o2==j)
      [c2,index]=sort(s2);
     break;
    end
    end
    [nn2, L2] = sstrain(nn2, temp_x, temp_y, opts);
    s2=sum(nn2.e.*nn2.e,2); 
    [c2,index]=sort(s2);
    temp_x=train_x_2(index(1:end),:);
    temp_y=train_y_2(index(1:end),:);
    [nn2,L2] = nntrain(nn2, temp_x, temp_y, opts);
    [er2, bad2] = nntest(nn2, test_x_2, test_y_2);
    er_2=er_2+er2; 
    % the third way 
    count_1=0;
    count_2=0;
    count_3=0;
    classone=[];
    classtwo=[];
    classthree=[];
    for j=1:576
       if train_y_3(j,1)==1;
           count_1=count_1+1;
           classone(count_1,1)=j;
       end
    end
    for j=1:576
       if train_y_3(j,1)==2;
           count_2=count_2+1;
           classtwo(count_2,1)=j;
       end
    end
    for j=1:576
       if train_y_3(j,1)==3;
           count_3=count_3+1;
           classthree(count_3,1)=j;
       end
    end
    r3=size(train_y_3,1);
    p3=size(test_y_3,1);
    train_y_3=[zeros(r3,my_max-1),train_y_3];
    for i=1:r3
        h=train_y_3(i,my_max);
        if (h<2)
        train_y_3(i,my_max)=0; 
        end
        train_y_3(i,h)=1;
        if(h<my_max)
        train_y_3(i,my_max)=0;
        end
    end
    test_y_3=[zeros(p3,my_max-1),test_y_3];
    for i=1:p3
        h=test_y_3(i,my_max);
        if (h<2)
        test_y_3(i,my_max)=0; 
        end
        test_y_3(i,h)=1;
        if(h<my_max)
        test_y_3(i,my_max)=0;
        end
    end
    % zscore is the standard deviation
    [train_x_3, mu, sigma] = zscore(train_x_3);
    test_x_3 = normalize(test_x_3, mu, sigma);
    %class one
    train_x1=train_x_3(classone(1:end),:);
    train_y1=train_y_3(classone(1:end),:);
    %class two
    train_x2=train_x_3(classtwo(1:end),:);
    train_y2=train_y_3(classtwo(1:end),:);
    %class three
    train_x3=train_x_3(classthree(1:end),:);
    train_y3=train_y_3(classthree(1:end),:);
    % train the first class
    [nn3,L] = sstrain(nn3, train_x1, train_y1, opts);
    s1=sum(nn3.e.*nn3.e,2);
    [c1,index_1]=sort(s1);
    lanbada_1=median(c1);
    temp_1=find(c1<lanbada_1);
    o1=size(temp_1,1);
    temp_x1=train_x1(index_1(1:o1),:);
    temp_y1=train_y1(index_1(1:o1),:);
    h1=lanbada_1;
    % train the second class
    [nn3,L] = sstrain(nn3, train_x2, train_y2, opts);
    s2=sum(nn3.e.*nn3.e,2);
    [c2,index_2]=sort(s2);
    lanbada_2=median(c2);
    temp_2=find(c2<lanbada_2);
    o2=size(temp_2,1);
    temp_x2=train_x2(index_2(1:o2),:);
    temp_y2=train_y2(index_2(1:o2),:);
    h2=lanbada_2;
    %train the third class
    [nn3,L] = sstrain(nn3, train_x3, train_y3, opts);
    s3=sum(nn3.e.*nn3.e,2);
    [c3,index_3]=sort(s3);
    lanbada_3=median(c3);
    temp_3=find(c3<lanbada_3);
    o3=size(temp_3,1);
    temp_x3=train_x3(index_3(1:o3),:);
    temp_y3=train_y3(index_3(1:o3),:);
    h3=lanbada_3;
    % update lanbada
    for i=1:100
    temp_x=[temp_x1;temp_x2;temp_x3];
    temp_y=[temp_y1;temp_y2;temp_y3];
    [nn3, L] = sstrain(nn3, temp_x, temp_y, opts);
    h1=h1*1.8;
    h2=h2*1.8;
    h3=h3*1.8;
    %update lanbada1
    temp_1=find(c1<h1);
    o1=size(temp_1,1);
    temp_x1=train_x1(index_1(1:o1),:);
    temp_y1=train_y1(index_1(1:o1),:);
    %update lanbada2
    temp_2=find(c2<h2);
    o2=size(temp_2,1);
    temp_x2=train_x2(index_2(1:o2),:);
    temp_y2=train_y2(index_2(1:o2),:);
    %update lanbada3
    temp_3=find(c3<h3);
    o3=size(temp_3,1);
    temp_x3=train_x3(index_3(1:o3),:);
    temp_y3=train_y3(index_3(1:o3),:);
    if (o1+o2+o3==576)
        break;
    end
    end
    % train the first class
    [nn3,L] = sstrain(nn3, train_x1, train_y1, opts);
    s1=sum(nn3.e.*nn3.e,2);
    [c1,index_1]=sort(s1);
    temp_x1=train_x1(index_1(1:end),:);
    temp_y1=train_y1(index_1(1:end),:);
    % train the second class
    [nn3,L] = sstrain(nn3, train_x2, train_y2, opts);
    s2=sum(nn3.e.*nn3.e,2);
    [c2,index_2]=sort(s2);
    temp_x2=train_x2(index_2(1:end),:);
    temp_y2=train_y2(index_2(1:end),:);
    %train the third class
    [nn3,L] = sstrain(nn3, train_x3, train_y3, opts);
    s3=sum(nn3.e.*nn3.e,2);
    [c3,index_3]=sort(s3);
    temp_x3=train_x3(index_3(1:end),:);
    temp_y3=train_y3(index_3(1:end),:);
    h3=lanbada_3;
    j=size(train_x3,1);
    % calculate the error rate 
    temp_x=[temp_x1;temp_x2;temp_x3];
    temp_y=[temp_y1;temp_y2;temp_y3];
    [nn3,L3] = nntrain(nn3, temp_x, temp_y, opts);
    [er3, bad3] = nntest(nn3, test_x_2, test_y_2);
    er_3=er_3+er3;
end
correct_1(num,1)=1-er_1/10
correct_2(num,1)=1-er_2/10
correct_3(num,1)=1-er_3/10
end
% ave1=mean(correct_1)
% standard_1=std(correct_1)
% ave2=mean(correct_2)
% standard_2=std(correct_2)
% ave3=mean(correct_3)
% standardm_3=std(correct_3)
% end


