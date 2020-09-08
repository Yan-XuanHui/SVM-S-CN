function [y]=svmval_categorical(x, xsup, w,b,kernel, kerneloption, dataSta, dotProduct,my_lambda)
    %x：要测试的数据，xsup：支持向量
    %dataSta：训练集的统计信息，dotProduct：训练集的点积

    %计算测试数据的核矩阵
    %[dotProduct, dataSta] = dotProductMatrix(x);
    %ps是核矩阵(测试数据与支持向量机xsup的核矩阵）
    ps = svmkernel_categorical(x, kernel, kerneloption, dataSta,dotProduct,my_lambda, xsup);
    
    y = ps*w+b;
    

