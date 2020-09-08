function [dataSta] = dataSta(xapp)

[r, dim] = size(xapp);		% r：行数，dim：列数（不含类别标签）

%%
%%统计各维字符出现的次数和频率（使用统计工具箱中的函数tabulate.m）
for i=1:dim 
    tt = xapp(:,i);
    B{1,i} = tabulate(tt);
end;

dataSta = B;