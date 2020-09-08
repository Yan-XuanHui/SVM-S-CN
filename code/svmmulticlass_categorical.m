function [xsup,w,b,nbsv,pos,obj]=svmmulticlass_categorical(x,y,my_lambda,nbclass,c,lambda,kernel,kerneloption,verbose, dataSta, dotProduct, ps)
%  nbclass: 类别数
%ps：高斯核矩阵

xsup=[];  % 3D matrices can not be used as numebr of SV changes
w=[];
b=[];
pos=[];
span=1;
alphainit = [];
qpsize=1000;
nbsv=zeros(1,nbclass);
%nbsuppvector=zeros(1,nbclass);
obj=0;
% one againt all:  每次把一个类别标签置为1，其余类别置为-1 
for i=1:nbclass
    
    yone=(y==i)+(y~=i)*-1;
    %if size(yone,1)>4000
   %     [xsupaux,waux,baux,posaux]=svmclassls(x,yone,c,epsilon,kernel,kerneloption,verbose,span,qpsize,alphainit);
   % else
   if nargin>11 
        [xsupaux,waux,baux,posaux,timeaux,alphaaux,objaux]=svmclass_categorical(x,yone,my_lambda, c,lambda,kernel,kerneloption,verbose,dataSta, dotProduct,span, alphainit,ps);
   else
       [xsupaux,waux,baux,posaux,timeaux,alphaaux,objaux]=svmclass_categorical(x,yone,my_lambda, c,lambda,kernel,kerneloption,verbose,dataSta, dotProduct,span, alphainit);
    end;
    
    nbsv(i)=length(posaux);
    xsup = [xsup;xsupaux];
    w=[w;waux];
    b=[b;baux];
    pos=[pos;posaux];
    obj=obj+objaux;
end;
