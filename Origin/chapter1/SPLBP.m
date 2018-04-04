%% �ô���Ϊ����BP���������ʶ��

%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%�������������ź�
load data1 c1
load data2 c2
load data3 c3
load data4 c4

% %�ĸ������źž���ϳ�һ������
% data(1:500,:)=c1(1:500,:);
% data(501:1000,:)=c2(1:500,:);
% data(1001:1500,:)=c3(1:500,:);
% data(1501:2000,:)=c4(1:500,:);

%��1��2000���������
k=rand(1,500); %�������һ��1*2000�ľ���
[m,n]=sort(k);  %mΪk�Ĵ�С�������У�nΪ��Ӧ��m���±�ʵ�����ɵ��������Ϊn

%�����������
input1 = c1(:,2:25);
input2 = c2(:,2:25);
input3 = c3(:,2:25);
input4 = c4(:,2:25);
output1=zeros(2000,4);
output2=zeros(2000,4);
output3=zeros(2000,4);
output4=zeros(2000,4);

%�������1ά���4ά
for i=1:500 %ѭ��-->��output�ж�Ӧ�����滻
    output1(i,:)=[1 0 0 0];
    output2(i,:)=[0 1 0 0];
    output3(i,:)=[0 0 1 0];
    output4(i,:)=[0 0 0 1];
end



%�����ȡ1500������Ϊѵ��������500������ΪԤ������
input_train_mid(1:375,:)=input1(n(1:375),:);%�����������n�������ѡ��1500����Ϊѵ������ת��
input_train_mid(376:750,:)=input2(n(1:375),:);
input_train_mid(751:1125,:)=input3(n(1:375),:);
input_train_mid(1126:1500,:)=input4(n(1:375),:);
output_train_mid(1:375,:)=output1(n(1:375),:);%��Ӧ��1500�����
output_train_mid(376:750,:)=output2(n(1:375),:);
output_train_mid(751:1125,:)=output3(n(1:375),:);
output_train_mid(1126:1500,:)=output4(n(1:375),:);

input_test_mid(1:125,:)=input1(n(376:500),:); %ʣ�µ�500��Ϊ����
input_test_mid(126:250,:)=input2(n(376:500),:);
input_test_mid(251:375,:)=input3(n(376:500),:);
input_test_mid(376:500,:)=input4(n(376:500),:);
output_test_mid(1:125,:)=output1(n(376:500),:); %��Ӧ��500���������
output_test_mid(126:250,:)=output2(n(376:500),:); 
output_test_mid(251:375,:)=output3(n(376:500),:); 
output_test_mid(376:500,:)=output4(n(376:500),:); 

output_1(1:125,:)=c1(n(376:500),1);
output_1(126:250,:)=c2(n(376:500),1);
output_1(251:375,:)=c3(n(376:500),1);
output_1(376:500,:)=c4(n(376:500),1);

k=rand(1,1500); %�������һ��1*2000�ľ���
[m,n]=sort(k);  %mΪk�Ĵ�С�������У�nΪ��Ӧ��m���±�ʵ�����ɵ��������Ϊn

input_train=input_train_mid(n,:)';
output_train=output_train_mid(n,:)';
input_test=input_test_mid';
output_test=output_test_mid';

%�������ݹ�һ��
[inputn,inputps]=mapminmax(input_train); %����μ�http://blog.csdn.net/u010480899/article/details/53485720

%% ����ṹ��ʼ��
innum=24; %L1
midnum=25; %L2
outnum=4; %L3
 

%Ȩֵ��ʼ��
w1=rands(midnum,innum); %%��һ���Weight
b1=rands(midnum,1); %%����ֵ����b1
w2=rands(midnum,outnum);
b2=rands(outnum,1);

w1_1=w1;w1_2=w1_1;
w2_1=w2;w2_2=w2_1;
b1_1=b1;b1_2=b1_1;
b2_1=b2;b2_2=b2_1;

%ѧϰ��
xite=0.1;
pace=0.8;
upace=1.5;
v=zeros(1,1500);

loopNumber=10; %ѭ������
I=zeros(1,midnum);
Iout=zeros(1,midnum);
FI=zeros(1,midnum);
dw1=zeros(innum,midnum);
db1=zeros(1,midnum);

%% ����ѵ��
E=zeros(1,loopNumber);
for ii=1:loopNumber
    E(ii)=0; %ѵ�����
    eznum=0;
    diff=0;
    cal_i=zeros(1,1500);
    for i=1:1:1500
       %% ����Ԥ����� 
        if v(i)~=2
            x=inputn(:,i);
            % ���������:Iout
            for j=1:1:midnum
                I(j)=inputn(:,i)'*w1(j,:)'+b1(j); %% �м�ֵ
                Iout(j)=1/(1+exp(-I(j)));
            end
            % ��������
            yn=w2'*Iout'+b2;

           %% Ȩֵ��ֵ����
            %�������
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
       %% ����Ԥ����� 
        x=inputn(:,cal_i(i));
        % ���������:Iout
        for j=1:1:midnum
            I(j)=x'*w1(j,:)'+b1(j); %% �м�ֵ
            Iout(j)=1/(1+exp(-I(j)));
        end
        % ��������
        yn=w2'*Iout'+b2;
        
       %% Ȩֵ��ֵ����
        %�������
        e=output_train(:,cal_i(i))-yn;     
        E(ii)=E(ii)+sum(abs(e));
        
        
        %����Ȩֵ�仯��
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j))); %%sigmod����
            FI(j)=S*(1-S); %%FI������sigmod�����ĵ���
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2)+e(3)*w2(j,3)+e(4)*w2(j,4));
            end
        end
          
        %%����Ȩֵ
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
 

%% ���������źŷ���
inputn_test=mapminmax('apply',input_test,inputps);
fore=zeros(4,500);
for ii=1:1
    for i=1:500%1500
        %���������
        for j=1:1:midnum
            I(j)=inputn_test(:,i)'*w1(j,:)'+b1(j);
            Iout(j)=1/(1+exp(-I(j)));
        end
        
        fore(:,i)=w2'*Iout'+b2;
    end
end

%% �������
%������������ҳ�������������
output_fore=zeros(1,500);
for i=1:500
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP����Ԥ�����
error=output_fore-output_1';

%����Ԥ�����������ʵ����������ķ���ͼ
figure(1)
plot(output_fore,'r')
hold on
plot(output_1','b')
legend('Ԥ���������','ʵ���������')

%�������ͼ
figure(2)
plot(error)
title('BP����������','fontsize',12)
xlabel('�����ź�','fontsize',12)
ylabel('�������','fontsize',12)

%print -dtiff -r600 1-4

k=zeros(1,4);  
%�ҳ��жϴ���ķ���������һ��
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

%�ҳ�ÿ��ĸ����
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

%��ȷ��
rightridio=(kk-k)./kk;
disp('��ȷ��')
disp(rightridio);