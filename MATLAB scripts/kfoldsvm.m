function [SVMModels,PSVMModels] = kfoldsvm(X,Ycon,numSubset,numGrid,hyperpts,cm)

k=size(X,1);
c = cvpartition(k,'KFold',numSubset);
SVMModels = cell(numSubset,1);
PSVMModels = cell(numSubset,1);

for i = 1:numSubset
    set = training(c,i);
    Xset = X(set,:);
    Yconset = Ycon(set);
    hyperparaopt = selectRbfPara(numGrid,hyperpts,Xset,Yconset);
    SVMModeli = fitcsvm(Xset,Yconset,'Standardize',true,'KernelFunction','rbf',...
        'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));     
      
    SVMModels{i}=SVMModeli;
    PSVMModeli = fitPosterior(SVMModeli); 
    PSVMModels{i} = PSVMModeli;
     
end