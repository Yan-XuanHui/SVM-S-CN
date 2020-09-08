%�����SVM_Categorical����
function [result] = SVM_Categorical_MultiClass_Fun(dataSet, K, KO1, KO2)  
%parameters:
%dataSet: Dataset's name��
%k:Kernel(gaussian��poly)��
%KO1: Parameter of SVM(degree or gamma)
%KO2: Parameter C of Gaussian Kernel
%%

filePath =  dataSet;
fprintf('Kernel:%s, KernelOption:%f\n', K, KO1);

[NUM,TXT,RAW]=xlsread(filePath);  %��ȡExcele���е����ݣ���Ϊ��arff�ļ�תΪExcel�ļ���
[m,n] = size(RAW); %n��������

%ѵ������������ռ70%��
m1 = round(m*0.7);
%���Լ���������ռ30%��
m2 = m - m1;

%����ҲҪת��Ϊ�ַ���ʽ
RAW = cellfun(@(x){num2str(x)},RAW);   %ת��Ϊ�ַ�����ʽ����Ϊ��Щ������������ʽҪ������Ϊ���ַ�

trainSet =  RAW(randperm(m, m1),:) ;  %ѵ���������ȡm1��
testSet = RAW(randperm(m, m2),:) ;

%ͳ�Ƹ�ά���ų��ֵĴ�����Ƶ��
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta����ͳ�Ƹ�ͳ����Ƶ��

%compute lambda
my_lambda = lambdaD(m, symbolSta, 2);   % computer bandwidth with MSE method

%׼�����Լ�
xapp = trainSet(:,1:n-1);  %��������ǩ
yapp_RAW = trainSet(:,n);
yapp_sta = tabulate(yapp_RAW);
yapp = zeros(m1,1);

[rows, cols] = size(yapp_sta);

%convert label to 1��2��3,...��multi-class��
for i=1:m1
    for j=1:rows
        if strcmp(yapp_sta{j}, yapp_RAW(i))
            yapp(i) = j;
        end;
    end;
end;
nbclass = rows;  %�����

%%
%ѵ����ʼ
kernel = K;
kerneloption = KO1;
lambda = 1e-7;  
C =  KO2;;     %bound on �������ճ���

if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %����ѵ�����ݵĵ��
else
    dotProduct = ones(1,1);
end;

verbose = 1;%�Ƿ���ʾ
%������ø�˹�ˣ���Ԥ�������˹�ˣ�����Ƕ���ʽ����ֱ���ã�(dotProduct+1)^degree
if( strcmp(kernel,'gaussian') )  
    ps = svmkernel_categorical(xapp, kernel, kerneloption, symbolSta, dotProduct, my_lambda); %��˹�˺�������Ҫ����dotProduct������� 
    %save gaussianKernel_Promoters ps;
    [xsup,w,w0,nbsv,pos,obj] = svmmulticlass_categorical(xapp, yapp, my_lambda, nbclass, C,lambda,kernel,kerneloption,verbose, symbolSta, dotProduct, ps);  %�����   
else
    [xsup,w,w0,nbsv,pos,obj] = svmmulticlass_categorical(xapp, yapp, my_lambda, nbclass, C,lambda,kernel,kerneloption,verbose, symbolSta, dotProduct);  %�����     
end;


%%
%���Կ�ʼ
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

[ypredapp,maxi] = svmmultival_Categorical(xtest,xsup,w,w0,nbsv,kernel,kerneloption, symbolSta, dotProduct, my_lambda);%����

%ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct);
%ypredapp = sign(ypredapp);    %ȡ���ķ���

correct_num = sum(ytest==ypredapp);
correct_percent = correct_num / m2  %׼ȷ��

%%
%���¼���F1_Measure
labels = unique(ytest);  %���
classNum = size(labels,1);  %���ٸ���
TP = zeros(classNum);
FP = zeros(classNum);
FN = zeros(classNum);
for i=1:classNum
   for j=1:m2
       if( ypredapp(j)==i &&  ytest(j)==i)   %��i��Ԥ��Ϊ�棬ʵ��Ϊ��
            TP(i) = TP(i) + 1; 
       end;
       if( ypredapp(j)==i && ytest(j)~=i )  %��i��Ԥ��Ϊ�棬ʵ��Ϊ��
            FP(i) = TP(i) + 1;  
       end;
       if( ytest(j)==i && ypredapp(j)~=i )  %��i��Ԥ��Ϊ�٣�ʵ��Ϊ��
            FN(i) = TP(i) + 1;  
       end;
   end;
end;
% num1 = sum(ytest==1);  %ʵ��Ϊ���������
% num2 = sum(ytest==-1); %ʵ��Ϊ�ٵ�������
Precision = zeros(classNum);
Recall = zeros(classNum);
F1_Score = zeros(classNum);
for i=1:classNum
  Precision(i) = TP(i)/(TP(i)+FP(i)); %����Ĳ�׼��  
  Recall(i) = TP(i)/(TP(i) + FN(i)); %������ٻ���  
  F1_Score(i) = 2*Precision(i)* Recall(i)/(Precision(i)+Recall(i));
end;

disp('Weight Avg. F1_Score:');
WF1_Score = 0;
for i=1:classNum
    WF1_Score = WF1_Score +  F1_Score(i)*sum(ytest==i);
end;
WF1_Score = WF1_Score / size(ytest,1);
%fprintf('Weight Avg. F1_Measure: %6.3f\n', WF1_Score);
result = WF1_Score;

