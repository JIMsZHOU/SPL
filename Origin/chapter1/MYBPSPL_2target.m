%% �ô���Ϊ����BP���������ʶ��

%% ��ջ�������
clc
clear

%% ѵ������Ԥ��������ȡ����һ��

%�������������ź�
load data1 c1
load data2 c2

%�ĸ������źž���ϳ�һ������
data(1:500,:)=c1(1:500,:);
data(501:1000,:)=c2(1:500,:);

%��1��2000���������
k=rand(1,1000); %�������һ��1*2000�ľ���
[m,n]=sort(k);  %mΪk�Ĵ�С�������У�nΪ��Ӧ��m���±�ʵ�����ɵ��������Ϊn

%�����������
input=data(:,2:25);
output1 =data(:,1);

%�������1ά���4ά
output=zeros(1000,1); %��ʼ��һ��2000*4�������
for i=1:1000 %ѭ��-->��output�ж�Ӧ�����滻
    if output1(i,:)==1
        output(i,:)=1;
    elseif output1(i,:)==2
        output(i,:)=2;
    end
end

%�����ȡ1500������Ϊѵ��������500������ΪԤ������
input_train=input(n(1:800),:)'; %�����������n�������ѡ��1500����Ϊѵ������ת��
output_train=output(n(1:800),:)'; %��Ӧ��1500�����

input_test=input(n(801:1000),:)'; %ʣ�µ�500��Ϊ����
output_test=output(n(801:1000),:)'; %��Ӧ��500���������

%�������ݹ�һ��
[inputn,inputps]=mapminmax(input_train); %����μ�http://blog.csdn.net/u010480899/article/details/53485720

%% ����ṹ��ʼ��
innum=24; %L1
midnum=25; %L2
outnum=2; %L3
 

%Ȩֵ��ʼ��
w1=rands(midnum,innum); %%��һ���Weight
b1=rands(midnum,1); %%����ֵ����b1
w2=rands(midnum,outnum);
b2=rands(outnum,1);


%ѧϰ��
xite=0.1;
pace=100;
updatepace=1;
loopNumber=20; %ѭ������
I=zeros(1,midnum);
Iout=zeros(1,midnum);
FI=zeros(1,midnum);
dw1=zeros(innum,midnum);
db1=zeros(1,midnum);
%% ����ѵ��
E=zeros(1,800);
for ii=1:loopNumber
    v=zeros(1,800);
    numofeasy=0;
    cal=zeros(1,800);
    for i=1:1:800
       %% ����Ԥ����� 
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
       %% ����Ԥ����� 
        x=inputn(:,cal(numofeasy));
        % ���������:Iout
        for j=1:1:midnum
            I(j)=x'*w1(j,:)'+b1(j); %% �м�ֵ
            Iout(j)=1/(1+exp(-I(j)));
        end
        % ��������
        yn=w2'*Iout'+b2;
        
       %% Ȩֵ��ֵ����
        %�������
        e=output_train(:,cal(numofeasy))-yn;     
        %����Ȩֵ�仯��
        dw2=e*Iout;
        db2=e';
        
        for j=1:1:midnum
            S=1/(1+exp(-I(j))); %%sigmod����
            FI(j)=S*(1-S); %%FI������sigmod�����ĵ���
        end      
        for k=1:1:innum
            for j=1:1:midnum
                dw1(k,j)=FI(j)*x(k)*(e(1)*w2(j,1)+e(2)*w2(j,2));
                db1(j)=FI(j)*(e(1)*w2(j,1)+e(2)*w2(j,2));
            end
        end
          
        %%����Ȩֵ
        w1=w1+xite*dw1';
        b1=b1+xite*db1';
        w2=w2+xite*dw2';
        b2=b2+xite*db2';
        
    end
    pace=pace*updatepace;
end
 

%% ���������źŷ���
inputn_test=mapminmax('apply',input_test,inputps);
fore=zeros(2,200);
for ii=1:1
    for i=1:200%1500
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
output_fore=zeros(2,200);
for i=1:200
    output_fore(i)=find(fore(:,i)==max(fore(:,i)));
end

%BP����Ԥ�����
error=output_fore-output1(n(801:1000))';

%����Ԥ�����������ʵ����������ķ���ͼ
figure(1)
plot(output_fore,'r')
hold on
plot(output1(n(801:1000))','b')
legend('Ԥ���������','ʵ���������')

%�������ͼ
figure(2)
plot(error)
title('BP����������','fontsize',12)
xlabel('�����ź�','fontsize',12)
ylabel('�������','fontsize',12)

%print -dtiff -r600 1-4

k=zeros(1,2);  
%�ҳ��жϴ���ķ���������һ��
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

%�ҳ�ÿ��ĸ����
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

%��ȷ��
rightridio=(kk-k)./kk;
disp('��ȷ��')
disp(rightridio);