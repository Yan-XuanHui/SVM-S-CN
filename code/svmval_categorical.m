function [y]=svmval_categorical(x, xsup, w,b,kernel, kerneloption, dataSta, dotProduct,my_lambda)
    %x��Ҫ���Ե����ݣ�xsup��֧������
    %dataSta��ѵ������ͳ����Ϣ��dotProduct��ѵ�����ĵ��

    %����������ݵĺ˾���
    %[dotProduct, dataSta] = dotProductMatrix(x);
    %ps�Ǻ˾���(����������֧��������xsup�ĺ˾���
    ps = svmkernel_categorical(x, kernel, kerneloption, dataSta,dotProduct,my_lambda, xsup);
    
    y = ps*w+b;
    

