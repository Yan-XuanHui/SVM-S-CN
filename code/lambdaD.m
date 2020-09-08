function lambda = lambdaD(N, symbolSta, method)
%�����ά�ĺ˴���lambda
%input:  
% N:���ݼ���������symbol:��ά���ŵ�Ƶ��ͳ��

[m, D] = size(symbolSta); %D�����ݼ�ά�ȣ�������ǩ��
lambda = zeros(1, D); %1*D����
for i = 1:D
    StaD = symbolSta{i};
    od = StaD(:,1);   %��һ��Ϊ���з����б�
    fod = StaD(:,3);  %������ΪƵ��*100
    [symbolNum, n] = size(od) ;
    fz = 0; %����
    fm = 0; %��ĸ
    sqr_frequency = 0; %����Ƶ��ƽ����
    for j =1: symbolNum     %����Ƶ��ƽ����
        fod1 = cell2mat(fod(j))/100.0; %����Ƶ��
        sqr_frequency =  sqr_frequency + fod1*fod1;
    end;
    if(method == 1)  %���ڽ�����֤�ķ��� 
        fz =  1 - sqr_frequency;
        fm = (N-1)*( sqr_frequency - 1.0/symbolNum );
        lambda(1, i) = fz/fm;   
    else %MSE����
        delta = 1 - sqr_frequency;
        fz =  symbolNum * delta;
        fm = symbolNum * (N + delta - N*delta) - delta;
        lambda(1, i) = fz/fm;       
    end;
end;


    


