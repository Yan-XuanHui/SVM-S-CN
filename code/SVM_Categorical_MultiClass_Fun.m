%多类别SVM_Categorical测试
function [result] = SVM_Categorical_MultiClass_Fun(dataSet, K, KO1, KO2)  
%parameters:
%dataSet: Dataset's name，
%k:Kernel(gaussian，poly)，
%KO1: Parameter of SVM(degree or gamma)
%KO2: Parameter C of Gaussian Kernel
%%

filePath =  dataSet;
fprintf('Kernel:%s, KernelOption:%f\n', K, KO1);

[NUM,TXT,RAW]=xlsread(filePath);  %读取Excele表中的数据（因为把arff文件转为Excel文件）
[m,n] = size(RAW); %n是属性数

%训练集样本数（占70%）
m1 = round(m*0.7);
%测试集样本数（占30%）
m2 = m - m1;

%数字也要转换为字符形式
RAW = cellfun(@(x){num2str(x)},RAW);   %转换为字符串形式，因为有些数据是数字形式要将其认为是字符

trainSet =  RAW(randperm(m, m1),:) ;  %训练集，随机取m1行
testSet = RAW(randperm(m, m2),:) ;

%统计各维符号出现的次数和频率
symbolSta = dataSta( RAW(:,1:n-1));   %dataSta函数统计各统符号频度

%compute lambda
my_lambda = lambdaD(m, symbolSta, 2);   % computer bandwidth with MSE method

%准备测试集
xapp = trainSet(:,1:n-1);  %不含类别标签
yapp_RAW = trainSet(:,n);
yapp_sta = tabulate(yapp_RAW);
yapp = zeros(m1,1);

[rows, cols] = size(yapp_sta);

%convert label to 1，2，3,...（multi-class）
for i=1:m1
    for j=1:rows
        if strcmp(yapp_sta{j}, yapp_RAW(i))
            yapp(i) = j;
        end;
    end;
end;
nbclass = rows;  %类别数

%%
%训练开始
kernel = K;
kerneloption = KO1;
lambda = 1e-7;  
C =  KO2;;     %bound on 拉格朗日乘子

if strcmp(kernel,'poly')
    [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda);  %计算训练数据的点积
else
    dotProduct = ones(1,1);
end;

verbose = 1;%是否显示
%如果是用高斯核，则预先算出高斯核，如果是多项式核则直接用：(dotProduct+1)^degree
if( strcmp(kernel,'gaussian') )  
    ps = svmkernel_categorical(xapp, kernel, kerneloption, symbolSta, dotProduct, my_lambda); %高斯核函数不需要参数dotProduct点积矩阵 
    %save gaussianKernel_Promoters ps;
    [xsup,w,w0,nbsv,pos,obj] = svmmulticlass_categorical(xapp, yapp, my_lambda, nbclass, C,lambda,kernel,kerneloption,verbose, symbolSta, dotProduct, ps);  %多类别   
else
    [xsup,w,w0,nbsv,pos,obj] = svmmulticlass_categorical(xapp, yapp, my_lambda, nbclass, C,lambda,kernel,kerneloption,verbose, symbolSta, dotProduct);  %多类别     
end;


%%
%测试开始
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

[ypredapp,maxi] = svmmultival_Categorical(xtest,xsup,w,w0,nbsv,kernel,kerneloption, symbolSta, dotProduct, my_lambda);%评价

%ypredapp = svmval_categorical(xtest,xsup,w,w0,kernel,kerneloption, symbolSta, dotProduct);
%ypredapp = sign(ypredapp);    %取数的符号

correct_num = sum(ytest==ypredapp);
correct_percent = correct_num / m2  %准确率

%%
%以下计算F1_Measure
labels = unique(ytest);  %类别
classNum = size(labels,1);  %多少个类
TP = zeros(classNum);
FP = zeros(classNum);
FN = zeros(classNum);
for i=1:classNum
   for j=1:m2
       if( ypredapp(j)==i &&  ytest(j)==i)   %第i类预测为真，实际为真
            TP(i) = TP(i) + 1; 
       end;
       if( ypredapp(j)==i && ytest(j)~=i )  %第i类预测为真，实际为假
            FP(i) = TP(i) + 1;  
       end;
       if( ytest(j)==i && ypredapp(j)~=i )  %第i类预测为假，实际为真
            FN(i) = TP(i) + 1;  
       end;
   end;
end;
% num1 = sum(ytest==1);  %实际为真的样本数
% num2 = sum(ytest==-1); %实际为假的样本数
Precision = zeros(classNum);
Recall = zeros(classNum);
F1_Score = zeros(classNum);
for i=1:classNum
  Precision(i) = TP(i)/(TP(i)+FP(i)); %正类的查准率  
  Recall(i) = TP(i)/(TP(i) + FN(i)); %正类的召回率  
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

