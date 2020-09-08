function [result]=SVM_Categoricla_Fun(dataSet, K, KO1, KO2)  
%parameters:
%dataSet: Dataset's name，
%k:Kernel(gaussian，poly)，
%KO1: Parameter of SVM(degree or gamma)
%KO2: Parameter C of Gaussian Kernel
%%

filePath =  dataSet;
fprintf('Kernel:%s, KernelOption:%f\n', K, KO1);

[NUM,TXT,RAW] = xlsread(filePath);  %read data from Excel file
%[NUM,TXT,RAW] = csvread('GermanCredit.xlsx')
[m,n] = size(RAW);

%size of train set（70%）
m1 = round(m*0.7);
%size of test set（30%）
m2 = m - m1;

%convert numerical data to symbolic data
RAW = cellfun(@(x){num2str(x)},RAW);     

trainSet =  RAW(randperm(m, m1),:) ;  % randomly generate the train set
testSet = RAW(randperm(m, m2),:) ;

%统计各维符号出现的次数和频率
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta: function of 统计各统符号频度

%compute lambda
my_lambda = lambdaD(m, symbolSta, 2); % computer bandwidth with MSE method

%weight = feature_weight(my_lambda, symbolSta);   %特征权重

%准备测试集
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
%训练开始
%kernel = 'gaussian';'poly'
kernel = K;
%kerneloption = 0.2;3
kerneloption = KO1;
lambda = 1e-7;  
C = KO2; %bound on 拉格朗日乘子
if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %计算训练数据的点积
else
    dotProduct = ones(1,1);
end;
[xsup,w,w0,pos,tps,alpha] = svmclass_categorical(xapp,yapp, my_lambda, C,lambda,kernel,kerneloption,1, symbolSta, dotProduct); 


%%
%测试开始
xtest = testSet(:,1:n-1);
ytest_RAW = testSet(:,n);%标签，n是维度
ytest = zeros(m2,1);
%把测试集标签转换为-1和1
for i=1:m2
    if strcmp(yapp_sta{1}, ytest_RAW(i))
        ytest(i) = 1;
    else
        ytest(i) = -1;
    end;
end;

ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct, my_lambda);
ypredapp = sign(ypredapp);    %取数的符号

correct_num = sum(ytest==ypredapp);
%disp('准确率:');
correct_percent = correct_num / m2;  %准确率

%计算查准率Precision指标
num1 = sum(ytest==1);  %实际为真的样本数
num2 = sum(ytest==-1); %实际为假的样本数
TP1 = 0 ; % 正类的正确肯定
FP1 = 0 ; % 正类的错误肯定
FN1 = 0 ; % 正类的错误否定
TP2 = 0 ; % 负类的正确肯定
FP2 = 0 ; % 负类的错误肯定
FN2 = 0 ; % 负类的错误否定
for i=1:m2
    if(ytest(i)==1 && ypredapp(i)==1)  %预测为真，实际为真 
        TP1 = TP1 + 1;
    end;
    if(ytest(i)==-1 && ypredapp(i)==1) %预测为真，实际为假
        FP1 = FP1 + 1;
    end;
    if(ytest(i)==1 && ypredapp(i)==-1) %预测为假，实际为真
        FN1 = FN1 + 1;
    end;   
    
    
    if(ytest(i)==-1 && ypredapp(i)==-1)  %预测为真，实际为真 
        TP2 = TP1 + 2;
    end;
    if(ytest(i)==1 && ypredapp(i)==-1) %预测为真，实际为假
        FP2 = FP2 + 1;
    end;
    if(ytest(i)==-1 && ypredapp(i)==1) %预测为假，实际为真
        FN2 = FN2 + 1;
    end;  
end;

Precision_1 = TP1/(TP1 + FP1);%正类的查准率
Recall_1 = TP1/(TP1 + FN1); %正类的召回率
F1_Score_1 = 2*Precision_1* Recall_1/(Precision_1+Recall_1);

Precision_2 = TP2/(TP2 + FP2);%正类的查准率
Recall_2 = TP2/(TP2 + FN2); %正类的召回率
F1_Score_2 = 2*Precision_2* Recall_2/(Precision_2+Recall_2);

F1_Score = ( F1_Score_1*sum(ytest==1) + F1_Score_2*sum(ytest==-1) )/(sum(ytest==1)+sum(ytest==-1));
%fprintf('加权F1_Measure: %6.3f\n', F1_Score);

result = F1_Score;

%[ps] = svmkernel_categorical(RAW , 'gaussian', 0.5, dataSta);  %计算高斯核矩阵
%[ps] = svmkernel_categorical(RAW , 'poly', 1, dataSta, dotProduct);  %计算多项式核矩阵