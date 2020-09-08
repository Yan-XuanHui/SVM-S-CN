function [dotProduct] = dotProductMatrix(xapp, symbolSta, my_lambda, xsup )
%%dotProductΪ�����ĵ������dataSta������Ƶ��ͳ�ƣ���i��cell��1���ǵ�iά�����ķ��ż����ڶ����Ǹ����ų��ֵĴ�������3����Ƶ��

if(nargin<4)
	xsup = xapp;
end;

lambda = my_lambda;

[r, dim] = size(xapp);		% r��������dim����������������ǩ��
[r1, dim1] = size(xsup);
B = cell(1,dim);            %����Ԫ������ÿ��cell��Ŷ�Ӧά���ַ������ݣ����ż������ִ�����Ƶ�ʣ�

%%
%%ͳ���ַ����ֵĴ�����Ƶ�ʣ�ʹ��ͳ�ƹ������еĺ���tabulate.m��
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
    [cd temp] = size(B{d});  %cdΪB{d}�����ά��(����ά���Է��Ÿ�����
    xy1(d) = ( cd-1)*lambda(d)^2/cd^2 + lambda(d)/cd*(1-lambda(d));
    xy2(d) = lambda(d)^2/cd^2 + lambda(d)/cd*(1-lambda(d)) + (1-lambda(d))^2 ;
end;
for i=1:r
    for j=1:r1
        for d=1:dim
             %[cd temp] = size(B{d});  %cdΪB{d}�����ά��(����ά���Է��Ÿ�����
             %xy(i,j,d)Ϊ��i���������j��������dά�ĵ��
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
%tt = cell2mat(xapp{1}(:,3))����ȡ�õ�1��cell��3�е����ݲ�ת��Ϊ����
