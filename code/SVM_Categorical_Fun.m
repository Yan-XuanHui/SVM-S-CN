function [result]=SVM_Categoricla_Fun(dataSet, K, KO1, KO2)  
%parameters:
%dataSet: Dataset's name��
%k:Kernel(gaussian��poly)��
%KO1: Parameter of SVM(degree or gamma)
%KO2: Parameter C of Gaussian Kernel
%%

filePath =  dataSet;
fprintf('Kernel:%s, KernelOption:%f\n', K, KO1);

[NUM,TXT,RAW] = xlsread(filePath);  %read data from Excel file
%[NUM,TXT,RAW] = csvread('GermanCredit.xlsx')
[m,n] = size(RAW);

%size of train set��70%��
m1 = round(m*0.7);
%size of test set��30%��
m2 = m - m1;

%convert numerical data to symbolic data
RAW = cellfun(@(x){num2str(x)},RAW);     

trainSet =  RAW(randperm(m, m1),:) ;  % randomly generate the train set
testSet = RAW(randperm(m, m2),:) ;

%ͳ�Ƹ�ά���ų��ֵĴ�����Ƶ��
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta: function of ͳ�Ƹ�ͳ����Ƶ��

%compute lambda
my_lambda = lambdaD(m, symbolSta, 2); % computer bandwidth with MSE method

%weight = feature_weight(my_lambda, symbolSta);   %����Ȩ��

%׼�����Լ�
xapp = trainSet(:,1:n-1);
yapp_RAW = trainSet(:,n);
yapp_sta = tabulate(yapp_RAW);
yapp = zeros(m1,1);

%convert label to -1 or 1
for i=1:m1
    if strcmp(yapp_sta{1}, yapp_RAW(i))
        yapp(i) = 1;
    else
        yapp(i) = -1;
    end;
end;


%%
%ѵ����ʼ
%kernel = 'gaussian';'poly'
kernel = K;
%kerneloption = 0.2;3
kerneloption = KO1;
lambda = 1e-7;  
C = KO2; %bound on �������ճ���
if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %����ѵ�����ݵĵ��
else
    dotProduct = ones(1,1);
end;
[xsup,w,w0,pos,tps,alpha] = svmclass_categorical(xapp,yapp, my_lambda, C,lambda,kernel,kerneloption,1, symbolSta, dotProduct); 


%%
%���Կ�ʼ
xtest = testSet(:,1:n-1);
ytest_RAW = testSet(:,n);%��ǩ��n��ά��
ytest = zeros(m2,1);
%�Ѳ��Լ���ǩת��Ϊ-1��1
for i=1:m2
    if strcmp(yapp_sta{1}, ytest_RAW(i))
        ytest(i) = 1;
    else
        ytest(i) = -1;
    end;
end;

ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct, my_lambda);
ypredapp = sign(ypredapp);    %ȡ���ķ���

correct_num = sum(ytest==ypredapp);
%disp('׼ȷ��:');
correct_percent = correct_num / m2;  %׼ȷ��

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

Precision_1 = TP1/(TP1 + FP1);%����Ĳ�׼��
Recall_1 = TP1/(TP1 + FN1); %������ٻ���
F1_Score_1 = 2*Precision_1* Recall_1/(Precision_1+Recall_1);

Precision_2 = TP2/(TP2 + FP2);%����Ĳ�׼��
Recall_2 = TP2/(TP2 + FN2); %������ٻ���
F1_Score_2 = 2*Precision_2* Recall_2/(Precision_2+Recall_2);

F1_Score = ( F1_Score_1*sum(ytest==1) + F1_Score_2*sum(ytest==-1) )/(sum(ytest==1)+sum(ytest==-1));
%fprintf('��ȨF1_Measure: %6.3f\n', F1_Score);

result = F1_Score;

%[ps] = svmkernel_categorical(RAW , 'gaussian', 0.5, dataSta);  %�����˹�˾���
%[ps] = svmkernel_categorical(RAW , 'poly', 1, dataSta, dotProduct);  %�������ʽ�˾���