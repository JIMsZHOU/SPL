

clc
clear

load mnist_uint8.mat

[trainnum,ftnum]=size(train_x);
targetnum=10;
testnum=size(test_x,1);

k=rand(1,trainnum);
[m,n]=sort(k);

input_train=train_x(n(1:trainnum),:)';
output_train=train_y(n(1:trainnum),:)';

[inputn,inputps]=mapminmax(input_train);

k=rand(1,testnum);
[m,n]=sort(k);

input_test=test_x(n(1:testnum),:)';
output_test=test_y(n(1:testnum),:)';
for i=1:1:testnum
	output_test_1=zeros(testnum,1);
	output_test_1(n(i))=find(test_y(i,:)==max(test_y(i,:)));
end

innum=ftnum;
midnum=ftnum+1;
outnum=targetnum;

w1=rands(midnum,innum);
b1=rands(midnum,1);
w2=rands(midnum,outnum);
b2=rands(outnum,1);

xite=0.1;
loopnum=10;
I=zeros(1,midnum);
Iout=zeros(1,midnum);
FI=zeros(1,midnum);
dw1=zeros(innum,midnum);
db1=zeros(1,midnum);

for ii=1:1:loopnum
	for i=1:1:trainnum
		x=inputn(:,i);
		y=output_train(:,i);
        
		for j=1:1:midnum
			I(i)=x'*w1(j,:)'+b1(j);
			Iout(j)=1/(1+exp(-I(j)));
        end
        yn=w2'*Iout'+b2;

        e=y-yn;

        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j))); %%sigmod函数
            FI(j)=S*(1-S); %%FI函数是sigmod函数的导数
        end

        for k=1:1:innum
            for j=1:1:midnum
                for p=1:1:outnum
                	we=0;
                	we=we+e(p)*w2(j,p);
                end
                dw1(k,j)=FI(j)*we;
                db1(j)=FI(j)*we;
            end
        end
          
        %%更新权值
        w1=w1_1+xite*dw1';
        b1=b1_1+xite*db1';
        w2=w2_1+xite*dw2';
        b2=b2_1+xite*db2';
        
    end
end

inputn_test=mapminmax('apply',input_test,inputps);
fore=zeros(targetnum,testnum);
for ii=1:1
    for i=1:1:testnum
        %隐含层输出
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end

output_fore=zeros(1:testnum);
for i=1:1:testnum
	output_fore(i)=find(fore(:,i)==max(fore(:,1)));
end

error=output_fore-output_test_1(n)';

%画出预测语音种类和实际语音种类的分类图
figure(1)
plot(output_fore,'r')
hold on
plot(output_test_1(n)','b')
legend('预测语音类别','实际语音类别')

%画出误差图
figure(2)
plot(error)
title('BP网络分类误差','fontsize',12)
xlabel('语音信号','fontsize',12)
ylabel('分类误差','fontsize',12)

k=zeros(1:targetnum);
for i=1:1:testnum
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

rightridio=(kk-k)./kk;
disp('正确率')
disp(rightridio);






