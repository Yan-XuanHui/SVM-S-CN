%SVM-S������ԣ�һ�����У�One-Versus-All OVA����ʽ��ÿһ�ν�����һ������Ϊ���࣬������Ϊ���࣬ÿ��ѵ��һ������������

clear;
[NUM,TXT,RAW]=xlsread('car1.xlsx');  %��ȡExcele���е����ݣ���Ϊ��arff�ļ�תΪExcel�ļ���
[m,n] = size(RAW);

%ѵ������������ռ80%��
m1 = round(m*0.8);
%m1=m;
%���Լ���������ռ20%��
m2 = m - m1;
%m2 = m;

%ת��Ϊ������
RAW = cellfun(@(x){num2str(x)},RAW);

trainSet =  RAW(randperm(m, m1),:) ;  %ѵ���������ȡm1��
testSet = RAW(randperm(m, m2),:) ;

%ͳ�Ƹ�ά���ų��ֵĴ�����Ƶ��
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta����ͳ�Ƹ�ͳ����Ƶ��

%����������ݵĺ˴���lambda
%my_lambda1 = lambdaD(m, symbolSta, 1);
my_lambda = lambdaD(m, symbolSta, 2);
%lambda = lambdaD(m, symbolSta);

%weight = feature_weight(my_lambda, symbolSta);   %����Ȩ��

%׼�����Լ�
xapp = trainSet(:,1:n-1);
yapp_RAW = trainSet(:,n);
yapp_sta = tabulate(yapp_RAW);
yapp = zeros(m1,1);

[rows, cols] = size(yapp_sta);
nbclass = rows;  %�����
%��ѵ������ǩת��Ϊ1��2��3,...�����ࣩ
for i=1:m1
    for j=1:rows
        if strcmp(yapp_sta{j}, yapp_RAW(i))
            yapp(i) = j;
        end;
    end;
end;

disp('The number of class:');
disp(nbclass);
xtest = testSet(:,1:n-1);
ytest_RAW = testSet(:,n);
ytest = zeros(m2,1);
%�Ѳ��Լ���ǩת��Ϊ1,2,3,...
for i=1:m2
    for j=1:rows
        if strcmp(yapp_sta{j}, ytest_RAW(i))
            ytest(i) = j;
        end;
    end;
end;

testclass = input('input test class:');
%��Ҫ���Ե���ת��Ϊ1������ת��Ϊ-1
yapp( yapp~=testclass ) = -1;
yapp( yapp==testclass ) = 1;  

ytest( ytest~=testclass ) = -1;
ytest( ytest==testclass ) = 1;   


%%
%ѵ����ʼ
kernel = 'gaussian';
%kernel = 'poly';
kerneloption = 0.1;
%kerneloption = 3;
lambda = 1e-7;  
C = 10;%bound on �������ճ���
if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %����ѵ�����ݵĵ��
else
    dotProduct = ones(1,1);
end;
[xsup,w,w0,pos,tps,alpha] = svmclass_categorical(xapp,yapp, my_lambda, C,lambda,kernel,kerneloption,1, symbolSta, dotProduct); 


%%
%���Կ�ʼ
% xtest = testSet(:,1:n-1);
% ytest_RAW = testSet(:,n);%��ǩ��n��ά��
% ytest = zeros(m2,1);
% %�Ѳ��Լ���ǩת��Ϊ-1��1
% for i=1:m2
%     if strcmp(yapp_sta{1}, ytest_RAW(i))
%         ytest(i) = 1;
%     else
%         ytest(i) = -1;
%     end;
% end;


%%
%���¼�������ָ��
ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct, my_lambda);
ypredapp = sign(ypredapp);    %ȡ���ķ���

correct_num = sum(ytest==ypredapp);
disp('׼ȷ��:');
correct_percent = correct_num / m2  %׼ȷ��

%�����׼��Precisionָ��
num1 = sum(ytest==1);  %ʵ��Ϊ���������
num2 = sum(ytest==-1); %ʵ��Ϊ�ٵ�������
TP1 = 0 ; % �������ȷ�϶�
FP1 = 0 ; % ����Ĵ���϶�
FN1 = 0 ; % ����Ĵ����
TP2 = 0 ; % �������ȷ�϶�
FP2 = 0 ; % ����Ĵ���϶�
FN2 = 0 ; % ����Ĵ����
for i=1:m2
    if(ytest(i)==1 && ypredapp(i)==1)  %Ԥ��Ϊ�棬ʵ��Ϊ�� 
        TP1 = TP1 + 1;
    end;
    if(ytest(i)==-1 && ypredapp(i)==1) %Ԥ��Ϊ�棬ʵ��Ϊ��
        FP1 = FP1 + 1;
    end;
    if(ytest(i)==1 && ypredapp(i)==-1) %Ԥ��Ϊ�٣�ʵ��Ϊ��
        FN1 = FN1 + 1;
    end;   
    
    
    if(ytest(i)==-1 && ypredapp(i)==-1)  %Ԥ��Ϊ�棬ʵ��Ϊ�� 
        TP2 = TP1 + 2;
    end;
    if(ytest(i)==1 && ypredapp(i)==-1) %Ԥ��Ϊ�棬ʵ��Ϊ��
        FP2 = FP2 + 1;
    end;
    if(ytest(i)==-1 && ypredapp(i)==1) %Ԥ��Ϊ�٣�ʵ��Ϊ��
        FN2 = FN2 + 1;
    end;  
end;

Precision_1 = TP1/(TP1+FP1);%����Ĳ�׼��
Recall_1 = TP1/(TP1 + FN1); %������ٻ���
F1_Score_1 = 2*Precision_1* Recall_1/(Precision_1+Recall_1);

Precision_2 = TP2/(TP2+FP2);%����Ĳ�׼��
Recall_2 = TP2/(TP2 + FN2); %������ٻ���
F1_Score_2 = 2*Precision_2* Recall_2/(Precision_2+Recall_2);

disp('��Ȩƽ��F1_Score:');
F1_Score = ( F1_Score_1*sum(ytest==1) + F1_Score_2*sum(ytest==-1) )/(sum(ytest==1)+sum(ytest==-1))


%[ps] = svmkernel_categorical(RAW , 'gaussian', 0.5, dataSta);  %�����˹�˾���
%[ps] = svmkernel_categorical(RAW , 'poly', 1, dataSta, dotProduct);  %�������ʽ�˾���