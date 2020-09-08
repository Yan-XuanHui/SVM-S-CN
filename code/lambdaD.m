function lambda = lambdaD(N, symbolSta, method)
%计算各维的核带宽lambda
%input:  
% N:数据集样本数；symbol:各维符号的频度统计

[m, D] = size(symbolSta); %D是数据集维度（不含标签）
lambda = zeros(1, D); %1*D矩阵
for i = 1:D
    StaD = symbolSta{i};
    od = StaD(:,1);   %第一列为所有符号列表
    fod = StaD(:,3);  %第三列为频度*100
    [symbolNum, n] = size(od) ;
    fz = 0; %分子
    fm = 0; %分母
    sqr_frequency = 0; %符号频率平方和
    for j =1: symbolNum     %计算频率平方和
        fod1 = cell2mat(fod(j))/100.0; %符号频率
        sqr_frequency =  sqr_frequency + fod1*fod1;
    end;
    if(method == 1)  %基于交叉验证的方法 
        fz =  1 - sqr_frequency;
        fm = (N-1)*( sqr_frequency - 1.0/symbolNum );
        lambda(1, i) = fz/fm;   
    else %MSE方法
        delta = 1 - sqr_frequency;
        fz =  symbolNum * delta;
        fm = symbolNum * (N + delta - N*delta) - delta;
        lambda(1, i) = fz/fm;       
    end;
end;


    


