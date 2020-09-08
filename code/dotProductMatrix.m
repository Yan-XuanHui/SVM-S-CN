function [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda, xsup )
%%dotProduct为样本的点积矩阵；dataSta：符号频率统计，第i个cell的1列是第i维特征的符号集，第二列是各符号出现的次数，第3列是频率

if(nargin<4)
	xsup = xapp;
end;

lambda = my_lambda;

[r, dim] = size(xapp);		% r：行数，dim：列数（不含类别标签）
[r1, dim1] = size(xsup);
B = cell(1,dim);            %定义元胞矩阵，每个cell存放对应维的字符集数据（符号集、出现次数和频率）

%%
%%统计字符出现的次数和频率（使用统计工具箱中的函数tabulate.m）
% for i=1:dim 
%     tt = xapp(:,i);
%     B{1,i} = tabulate(tt);
% end;
B = symbolSta;

%%

xy = zeros(r,r1,dim);
xy1 = zeros(dim);
xy2 = zeros(dim);
for d=1:dim
    [cd temp] = size(B{d});  %cd为B{d}矩阵的维度(即该维属性符号个数）
    xy1(d) = ( cd-1)*lambda(d)^2/cd^2 + lambda(d)/cd*(1-lambda(d));
    xy2(d) = lambda(d)^2/cd^2 + lambda(d)/cd*(1-lambda(d)) + (1-lambda(d))^2 ;
end;
for i=1:r
    for j=1:r1
        for d=1:dim
             %[cd temp] = size(B{d});  %cd为B{d}矩阵的维度(即该维属性符号个数）
             %xy(i,j,d)为第i个样本与第j个样本第d维的点积
             %xy1 = ( cd-1)*lamda^2/cd^2 + lamda/cd*(1-lamda);
             %xy2 = lamda^2/cd^2 + lamda/cd*(1-lamda) + (1-lamda)^2 ;
             flag = 0;
             if strcmp(xapp(i,d),xsup(j,d)) 
                 flag = 1;
             end;   
             xy(i, j, d) = xy1(d) + xy2(d)*flag;
        end;
    end;
end;

%%
xy_t = zeros(r,r1);
for i=1:r
    for j=1:r1
        xy_t(i,j) = sum( xy(i,j,:) );
    end;
end;
dotProduct = xy_t;
%disp('dotProduct finish');
%dataSta = B;
%tt = cell2mat(xapp{1}(:,3))可以取得第1个cell第3列的数据并转化为矩阵
