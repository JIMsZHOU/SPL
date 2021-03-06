%% 该代码为基于BP网络的语言识别

%% 清空环境变量
clc
clear

%% 训练数据预测数据提取及归一化

%下载四类语音信号
load data1 c1
load data2 c2

%四个特征信号矩阵合成一个矩阵
data(1:500,:)=c1(1:500,:);
data(501:1000,:)=c2(1:500,:);

%从1到2000间随机排序
k=rand(1,1000); %随机生成一个1*2000的矩阵
[m,n]=sort(k);  %m为k的从小到大排列，n为对应的m的下表／实际生成的随机数列为n

%输入输出数据
input=data(:,2:25);
output1 =data(:,1);

%把输出从1维变成4维
output=zeros(1000,1); %初始化一个2000*4的零矩阵
for i=1:1000 %循环-->将output中对应的零替换
    if output1(i,:)==1
        output(i,:)=1;
    elseif output1(i,:)==2
        output(i,:)=2;
    end
end

%随机提取1500个样本为训练样本，500个样本为预测样本
input_train=input(n(1:800),:)'; %利用随机数列n，来随机选择1500行作为训练，并转置
output_train=output(n(1:800),:)'; %对应的1500个输出

input_test=input(n(801:1000),:)'; %剩下的500作为测试
output_test=output(n(801:1000),:)'; %对应的500个输出测试

%输入数据归一化
[inputn,inputps]=mapminmax(input_train); %详情参见http://blog.csdn.net/u010480899/article/details/53485720

%% 网络结构初始化
innum=24; %L1
midnum=25; %L2
outnum=2; %L3
 

%权值初始化
w1=rands(midnum,innum); %%第一层的Weight
b1=rands(midnum,1); %%修正值参数b1
w2=rands(midnum,outnum);
b2=rands(outnum,1);


%学习率
xite=0.1;
pace=100;
updatepace=1;
loopNumber=20; %循环次数
I=zeros(1,midnum);
Iout=zeros(1,midnum);
FI=zeros(1,midnum);
dw1=zeros(innum,midnum);
db1=zeros(1,midnum);
%% 网络训练
E=zeros(1,800);
for ii=1:loopNumber
    v=zeros(1,800);
    numofeasy=0;
    cal=zeros(1,800);
    for i=1:1:800
       %% 网络预测输出 
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
        E(i)=sum(abs(e));
        
        if E(i)<pace
            v(i)=1;
            numofeasy=numofeasy+1;
            cal(numofeasy)=i;
        else
            v(i)=0;
        end
    end
    for i=1:1:numofeasy
       %% 网络预测输出 
        x=inputn(:,cal(numofeasy));
        % 隐含层输出:Iout
        for j=1:1:midnum
            I(j)=x'*w1(j,:)'+b1(j); %% 中间值
            Iout(j)=1/(1+exp(-I(j)));
        end
        % 输出层输出
        yn=w2'*Iout'+b2;
        
       %% 权值阀值修正
        %计算误差
        e=output_train(:,cal(numofeasy))-yn;     
        %计算权值变化率
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j))); %%sigmod函数
            FI(j)=S*(1-S); %%FI函数是sigmod函数的导数
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));
            end
        end
          
        %%更新权值
        w1=w1+xite*dw1';
        b1=b1+xite*db1';
        w2=w2+xite*dw2';
        b2=b2+xite*db2';
        
    end
    pace=pace*updatepace;
end
 

%% 语音特征信号分类
inputn_test=mapminmax('apply',input_test,inputps);
fore=zeros(2,200);
for ii=1:1
    for i=1:200%1500
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
output_fore=zeros(2,200);
for i=1:200
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP网络预测误差
error=output_fore-output1(n(801:1000))';

%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output1(n(801:1000))','b')
legend('预测语音类别','实际语音类别')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('语音信号','fontsize',12)
ylabel('分类误差','fontsize',12)

%print -dtiff -r600 1-4

k=zeros(1,2);  
%找出判断错误的分类属于哪一类
for i=1:200
    if error(i)~=0
        [b,c]=max(output_test(:,i));
        switch c
            case 1 
                k(1)=k(1)+1;
            case 2 
                k(2)=k(2)+1;
        end
    end
end

%找出每类的个体和
kk=zeros(1,2);
for i=1:200
    [b,c]=max(output_test(:,i));
    switch c
        case 1
            kk(1)=kk(1)+1;
        case 2
            kk(2)=kk(2)+1;
    end
end

%正确率
rightridio=(kk-k)./kk;
disp('正确率')
disp(rightridio);