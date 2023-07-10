clearvars
close all
% Step 1:%%% Construct 5x187x100 matrix as mat1.mat:100 is no.of subjects.
% Step 2: Construct label file 
num_rowsfromcsv=34
nn=num_rowsfromcsv;
load('data_cnreho.mat')
load('data_mcireho.mat')
len=40
COVlist_mci=covariances_ammu(data_mci);
COVlist_cn=covariances_ammu(data_cn);

record_labels_mci=ones(len,1);
record_labels_cn=2*ones(len,1);

record_labels_big=cat(1,record_labels_mci,record_labels_cn); %104x1
     
COVBIGlist=cat(3,COVlist_mci,COVlist_cn); %combined 5x5 covariance matrices for 47+47 subjects., ie., 5x5x104 matrix
BIGlist=E1gen(COVBIGlist,length(record_labels_big));


n1_idx= find(record_labels_big==1);
n2_idx= find(record_labels_big==2);

num=min([length(n1_idx) length(n2_idx)]);

y1=datasample(n1_idx,num);
y2=datasample(n2_idx,num);

x=cat(1,y1,y2);
% random_array=randperm(num*5,num*5);
%     x=x(random_array);

BIGlist_unb=BIGlist(x,:);
record_labels_unb=record_labels_big(x);

indices=crossvalind('Kfold',length(record_labels_unb),10);

% plot_2d_visualization(BIGlist_unb,record_labels_unb);
figure;
acc=zeros(10,1);
for i=1:10
    test=(indices==i);
    train=~test;
    BIGtest=BIGlist_unb(test,:);
    BIGtrain=BIGlist_unb(train,:);
    Ytrain=record_labels_unb(train);
    trueYtest=record_labels_unb(test);

    Model=fitcensemble(BIGtrain,Ytrain, 'Method','Bag','NumLearningCycles',497);

    Ytest=predict(Model,BIGtest);
    acc(i) = 100*mean(Ytest==trueYtest);     
end

result_E1 = mean(acc)

%%%%%%%%%%%%%%%%%%
BIGlist=E2gen(COVBIGlist,length(record_labels_big),nn);

BIGlist_unb=BIGlist(x,:);
% record_labels_unb=record_labels_big(x);
% plot_2d_visualization(BIGlist_unb,record_labels_unb);
figure;
indices=crossvalind('Kfold',length(record_labels_unb),10);
acc=zeros(10,1);
for i=1:10
    test=(indices==i);
    train=~test;
    BIGtest=BIGlist_unb(test,:);
    BIGtrain=BIGlist_unb(train,:);
    Ytrain=record_labels_unb(train);
    trueYtest=record_labels_unb(test);

    Model=fitcensemble(BIGtrain,Ytrain, 'Method','Bag','NumLearningCycles',497);

    Ytest=predict(Model,BIGtest);
    acc(i) = 100*mean(Ytest==trueYtest);     
end

result_E2 = mean(acc)

%%%%%%%%%%%%%%%%%% check this.................
BIGlist=MFgen(COVBIGlist,length(record_labels_big));

BIGlist_unb=BIGlist(x,:);
% record_labels_unb=record_labels_big(x);

% plot_2d_visualization(BIGlist_unb,record_labels_unb);
figure;
indices=crossvalind('Kfold',length(record_labels_unb),10);
acc=zeros(10,1);
for i=1:10
    test=(indices==i);
    train=~test;
    BIGtest=BIGlist_unb(test,:);
    BIGtrain=BIGlist_unb(train,:);
    Ytrain=record_labels_unb(train);
    trueYtest=record_labels_unb(test);

    Model=fitcensemble(BIGtrain,Ytrain, 'Method','Bag','NumLearningCycles',497);

    Ytest=predict(Model,BIGtest);
    acc(i) = 100*mean(Ytest==trueYtest);     
end

result_MF = mean(acc)
%%%%%%%%%%%%%

COVlist_unb=COVBIGlist(:,:,x);
% record_labels_unb=record_labels_big(x);
% plot_2d_visualization(COVlist_unb,record_labels_unb);

indices=crossvalind('Kfold',length(record_labels_unb),10);
acc=zeros(10,1);
for i=1:10
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

result_ts = mean(acc)



