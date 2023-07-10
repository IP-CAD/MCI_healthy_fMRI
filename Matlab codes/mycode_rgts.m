len=47
COVlist_mci=covariances_ammu(data_mci);
COVlist_cn=covariances_ammu(data_cn);

record_labels_mci=ones(len,1);
record_labels_cn=2*ones(len,1);

record_labels_big=cat(1,record_labels_mci,record_labels_cn); %104x1
     
COVBIGlist=cat(3,COVlist_mci,COVlist_cn); %combined 5x5 covariance matrices for 52+52 subjects., ie., 5x5x104 matrix
% BIGlist=MFgen(COVBIGlist,length(record_labels_big));


n1_idx= find(record_labels_big==1);
n2_idx= find(record_labels_big==2);

num=min([length(n1_idx) length(n2_idx)]);

y1=datasample(n1_idx,num);
y2=datasample(n2_idx,num);

x=cat(1,y1,y2);
% random_array=randperm(num*5,num*5);
%     x=x(random_array);

% BIGlist_unb=BIGlist(x,:);
COVlist_unb=COVBIGlist(:,:,x);
record_labels_unb=record_labels_big(x);

indices=crossvalind('Kfold',length(record_labels_unb),10);
acc=zeros(10,1);
for i=1:10
    i
    test=(indices==i);
    train=~test;
%     BIGtest=BIGlist_unb(test,:);
%     BIGtrain=BIGlist_unb(train,:);
    COVtest=COVlist_unb(:,:,test);
    COVtrain=COVlist_unb(:,:,train);
    Ytrain=record_labels_unb(train);
    trueYtest=record_labels_unb(test);
    Ytest=ts_ensemble(COVtest,COVtrain,Ytrain);
%     Model=fitcensemble(BIGtrain,Ytrain, 'Method','Bag','NumLearningCycles',497);

%     Ytest=predict(Model,BIGtest);
    acc(i) = 100*mean(Ytest==trueYtest);     
end

result4 = mean(acc);