%SVM-S多类测试，一对所有（One-Versus-All OVA）方式，每一次将其中一个类作为正类，其余作为负类，每次训练一个二分类器。

clear;
[NUM,TXT,RAW]=xlsread('car1.xlsx');  %读取Excele表中的数据（因为把arff文件转为Excel文件）
[m,n] = size(RAW);

%训练集样本数（占80%）
m1 = round(m*0.8);
%m1=m;
%测试集样本数（占20%）
m2 = m - m1;
%m2 = m;

%转换为符号型
RAW = cellfun(@(x){num2str(x)},RAW);

trainSet =  RAW(randperm(m, m1),:) ;  %训练集，随机取m1行
testSet = RAW(randperm(m, m2),:) ;

%统计各维符号出现的次数和频率
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta函数统计各统符号频度

%计算符号数据的核带宽lambda
%my_lambda1 = lambdaD(m, symbolSta, 1);
my_lambda = lambdaD(m, symbolSta, 2);
%lambda = lambdaD(m, symbolSta);

%weight = feature_weight(my_lambda, symbolSta);   %特征权重

%准备测试集
xapp = trainSet(:,1:n-1);
yapp_RAW = trainSet(:,n);
yapp_sta = tabulate(yapp_RAW);
yapp = zeros(m1,1);

[rows, cols] = size(yapp_sta);
nbclass = rows;  %类别数
%把训练集标签转换为1，2，3,...（多类）
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
%把测试集标签转换为1,2,3,...
for i=1:m2
    for j=1:rows
        if strcmp(yapp_sta{j}, ytest_RAW(i))
            ytest(i) = j;
        end;
    end;
end;

testclass = input('input test class:');
%将要测试的类转换为1，其余转换为-1
yapp( yapp~=testclass ) = -1;
yapp( yapp==testclass ) = 1;  

ytest( ytest~=testclass ) = -1;
ytest( ytest==testclass ) = 1;   


%%
%训练开始
kernel = 'gaussian';
%kernel = 'poly';
kerneloption = 0.1;
%kerneloption = 3;
lambda = 1e-7;  
C = 10;%bound on 拉格朗日乘子
if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %计算训练数据的点积
else
    dotProduct = ones(1,1);
end;
[xsup,w,w0,pos,tps,alpha] = svmclass_categorical(xapp,yapp, my_lambda, C,lambda,kernel,kerneloption,1, symbolSta, dotProduct); 


%%
%测试开始
% xtest = testSet(:,1:n-1);
% ytest_RAW = testSet(:,n);%标签，n是维度
% ytest = zeros(m2,1);
% %把测试集标签转换为-1和1
% for i=1:m2
%     if strcmp(yapp_sta{1}, ytest_RAW(i))
%         ytest(i) = 1;
%     else
%         ytest(i) = -1;
%     end;
% end;


%%
%以下计算评价指标
ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct, my_lambda);
ypredapp = sign(ypredapp);    %取数的符号

correct_num = sum(ytest==ypredapp);
disp('准确率:');
correct_percent = correct_num / m2  %准确率

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

Precision_1 = TP1/(TP1+FP1);%正类的查准率
Recall_1 = TP1/(TP1 + FN1); %正类的召回率
F1_Score_1 = 2*Precision_1* Recall_1/(Precision_1+Recall_1);

Precision_2 = TP2/(TP2+FP2);%正类的查准率
Recall_2 = TP2/(TP2 + FN2); %正类的召回率
F1_Score_2 = 2*Precision_2* Recall_2/(Precision_2+Recall_2);

disp('加权平均F1_Score:');
F1_Score = ( F1_Score_1*sum(ytest==1) + F1_Score_2*sum(ytest==-1) )/(sum(ytest==1)+sum(ytest==-1))


%[ps] = svmkernel_categorical(RAW , 'gaussian', 0.5, dataSta);  %计算高斯核矩阵
%[ps] = svmkernel_categorical(RAW , 'poly', 1, dataSta, dotProduct);  %计算多项式核矩阵