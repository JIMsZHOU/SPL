%% 该代码为基于BP网络的语言识别

%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%下载四类语音信号
load data1 c1
load data2 c2
load data3 c3
load data4 c4

% %四个特征信号矩阵合成一个矩阵
% data(1:500,:)=c1(1:500,:);
% data(501:1000,:)=c2(1:500,:);
% data(1001:1500,:)=c3(1:500,:);
% data(1501:2000,:)=c4(1:500,:);

%从1到2000间随机排序
k=rand(1,500); %随机生成一个1*2000的矩阵
[m,n]=sort(k);  %m为k的从小到大排列，n为对应的m的下表／实际生成的随机数列为n

%输入输出数据
input1 = c1(:,2:25);
input2 = c2(:,2:25);
input3 = c3(:,2:25);
input4 = c4(:,2:25);
output1=zeros(2000,4);
output2=zeros(2000,4);
output3=zeros(2000,4);
output4=zeros(2000,4);

%把输出从1维变成4维
for i=1:500 %循环-->将output中对应的零替换
    output1(i,:)=[1 0 0 0];
    output2(i,:)=[0 1 0 0];
    output3(i,:)=[0 0 1 0];
    output4(i,:)=[0 0 0 1];
end



%随机提取1500个样本为训练样本，500个样本为预测样本
input_train_mid(1:375,:)=input1(n(1:375),:);%利用随机数列n，来随机选择1500行作为训练，并转置
input_train_mid(376:750,:)=input2(n(1:375),:);
input_train_mid(751:1125,:)=input3(n(1:375),:);
input_train_mid(1126:1500,:)=input4(n(1:375),:);
output_train_mid(1:375,:)=output1(n(1:375),:);%对应的1500个输出
output_train_mid(376:750,:)=output2(n(1:375),:);
output_train_mid(751:1125,:)=output3(n(1:375),:);
output_train_mid(1126:1500,:)=output4(n(1:375),:);

input_test_mid(1:125,:)=input1(n(376:500),:); %剩下的500作为测试
input_test_mid(126:250,:)=input2(n(376:500),:);
input_test_mid(251:375,:)=input3(n(376:500),:);
input_test_mid(376:500,:)=input4(n(376:500),:);
output_test_mid(1:125,:)=output1(n(376:500),:); %对应的500个输出测试
output_test_mid(126:250,:)=output2(n(376:500),:); 
output_test_mid(251:375,:)=output3(n(376:500),:); 
output_test_mid(376:500,:)=output4(n(376:500),:); 

output_1(1:125,:)=c1(n(376:500),1);
output_1(126:250,:)=c2(n(376:500),1);
output_1(251:375,:)=c3(n(376:500),1);
output_1(376:500,:)=c4(n(376:500),1);

k=rand(1,1500); %随机生成一个1*2000的矩阵
[m,n]=sort(k);  %m为k的从小到大排列，n为对应的m的下表／实际生成的随机数列为n

input_train=input_train_mid(n,:)';
output_train=output_train_mid(n,:)';
input_test=input_test_mid';
output_test=output_test_mid';

%输入数据归一化
[inputn,inputps]=mapminmax(input_train); %详情参见http://blog.csdn.net/u010480899/article/details/53485720

%% 网络结构初始化
innum=24; %L1
midnum=25; %L2
outnum=4; %L3
 

%权值初始化
w1=rands(midnum,innum); %%第一层的Weight
b1=rands(midnum,1); %%修正值参数b1
w2=rands(midnum,outnum);
b2=rands(outnum,1);

w1_1=w1;w1_2=w1_1;
w2_1=w2;w2_2=w2_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%学习率
xite=0.1;
pace=0.8;
upace=1.5;
v=zeros(1,1500);

loopNumber=10; %循环次数
I=zeros(1,midnum);
Iout=zeros(1,midnum);
FI=zeros(1,midnum);
dw1=zeros(innum,midnum);
db1=zeros(1,midnum);

%% 网络训练
E=zeros(1,loopNumber);
for ii=1:loopNumber
    E(ii)=0; %训练误差
    eznum=0;
    diff=0;
    cal_i=zeros(1,1500);
    for i=1:1:1500
       %% 网络预测输出 
        if v(i)~=2
            x=inputn(:,i);
            % 隐含层输出:Iout
            for j=1:1:midnum
                I(j)=inputn(:,i)'*w1(j,:)'+b1(j); %% 中间值
                Iout(j)=1/(1+exp(-I(j)));
            end
            % 输出层输出
            yn=w2'*Iout'+b2;

           %% 权值阀值修正
            %计算误差
            e=output_train(:,i)-yn;     
            E(ii)=E(ii)+sum(abs(e));            
            diff=sum(abs(e))/4;
            if diff<pace
                v(i)=1;
                eznum=eznum+1;
                cal_i(eznum)=i;
            else
                v(i)=0;
            end
        end
    end
    
    
    
    for i=1:1:eznum
       %% 网络预测输出 
        x=inputn(:,cal_i(i));
        % 隐含层输出:Iout
        for j=1:1:midnum
            I(j)=x'*w1(j,:)'+b1(j); %% 中间值
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %计算误差
        e=output_train(:,cal_i(i))-yn;     
        E(ii)=E(ii)+sum(abs(e));
        
        
        %计算权值变化率
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j))); %%sigmod函数
            FI(j)=S*(1-S); %%FI函数是sigmod函数的导数
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
            end
        end
          
        %%更新权值
        w1=w1_1+xite*dw1';
        b1=b1_1+xite*db1';
        w2=w2_1+xite*dw2';
        b2=b2_1+xite*db2';
        
        w1_2=w1_1;w1_1=w1;
        w2_2=w2_1;w2_1=w2;
        b1_2=b1_1;b1_1=b1;
        b2_2=b2_1;b2_1=b2;
        
        v(cal_i(i))=2;
    end
    pace=pace*upace;
end
 

%% 语音特征信号分类
inputn_test=mapminmax('apply',input_test,inputps);
fore=zeros(4,500);
for ii=1:1
    for i=1:500%1500
        %隐含层输出
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end

%% 结果分析
%根据网络输出找出数据属于哪类
output_fore=zeros(1,500);
for i=1:500
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP网络预测误差
error=output_fore-output_1';

%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output_1','b')
legend('预测语音类别','实际语音类别')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('语音信号','fontsize',12)
ylabel('分类误差','fontsize',12)

%print -dtiff -r600 1-4

k=zeros(1,4);  
%找出判断错误的分类属于哪一类
for i=1:500
    if error(i)~=0
        [b,c]=max(output_test(:,i));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
            case 3 
                k(3)=k(3)+1;
            case 4 
                k(4)=k(4)+1;
        end
    end
end

%找出每类的个体和
kk=zeros(1,4);
for i=1:500
    [b,c]=max(output_test(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
        case 3
            kk(3)=kk(3)+1;
        case 4
            kk(4)=kk(4)+1;
    end
end

%正确率
rightridio=(kk-k)./kk;
disp('正确率')
disp(rightridio);