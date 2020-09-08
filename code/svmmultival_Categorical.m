function [ypred,maxi,ypredMat]=svmmultival_Categorical(x,xsup,w,b,nbsv,kernel,kerneloption, symbolSta, dotProduct,my_lambda)

% USAGE ypred=svmmultival(x,xsup,w,b,nbsv,kernel,kerneloption)
% 

[n1,n2]=size(x);
nbclass=length(nbsv);
y=zeros(n1,nbclass);
nbsv=[0 nbsv];
aux=cumsum(nbsv);
for i=1:nbclass
    xsupaux=xsup(aux(i)+1:aux(i)+nbsv(i+1),:);
	waux=w(aux(i)+1:aux(i)+nbsv(i+1));
	baux=b(i);
	ypred(:,i)= svmval_categorical(x,xsupaux,waux,baux,kernel,kerneloption, symbolSta, dotProduct,my_lambda);
                %svmval_categorical(x, xsup,   w,   b,  kernel, kerneloption, dataSta, dotProduct,my_lambda)
    
end;
ypredMat=ypred;
[maxi,ypred] = max(ypred');%ypred'各列最大值和行号，matlab数组的下标是从1号开始
maxi=maxi';
ypred=ypred';
