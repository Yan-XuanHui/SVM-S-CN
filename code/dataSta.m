function [dataSta] = dataSta(xapp)

[r, dim] = size(xapp);		% r��������dim����������������ǩ��

%%
%%ͳ�Ƹ�ά�ַ����ֵĴ�����Ƶ�ʣ�ʹ��ͳ�ƹ������еĺ���tabulate.m��
for i=1:dim 
    tt = xapp(:,i);
    B{1,i} = tabulate(tt);
end;

dataSta = B;