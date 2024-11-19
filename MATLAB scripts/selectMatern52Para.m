function para = selectMatern52Para(X,Ycon)

d = 1;
lb = 1e-5;
ub = 1e3;

k = 10*d;
paralhs = lhsdesign(k, d); % design points
% Xlhs = gridsamp([0, 0;1, 1], 3);
paraset = zeros(k, d);
for i = 1:k
    paraset(i,:) = lb + (ub - lb).*paralhs(i,:);
end

c = cvpartition(size(X,1),'KFold',10);
lossset = zeros(k,1);
for i=1:k
    value = paraset(i,:);
    passkMatern52Para(value);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'CVPartition',c,'KernelFunction','mymatern52');
    lossset(i) = kfoldLoss(SVMModel,'LossFun','classiferror');
end

gprMdl = fitrgp(paraset,lossset);


%% bayesian optimization for hyperparameters
iter = 1;
while iter < 200
    
    fmin = min(lossset);
    testNum = 10*d;
    testlhs = lhsdesign(testNum, d); 
    testset = zeros(testNum, d);
    testEI = zeros(testNum, 1);
    for i = 1:testNum
        testset(i,:) = lb + (ub - lb).*testlhs(i,:);
        testpt = testset(i,:);
        testEI(i) = EI(fmin,testpt,gprMdl);
    end
    [~,I] = max(testEI);
    testini = testset(I,:);
    
    [paraAdd, EIAdd] = fmincon(@(x) -EI(fmin,x,gprMdl), testini,[], [], [], [], lb, ub);

    % update paraset and lossset
    paraset(end+1,:) = paraAdd;
    value = paraAdd;
    passkMatern52Para(value);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'CVPartition',c,'KernelFunction','mymatern52');
    lossAdd = kfoldLoss(SVMModel,'LossFun','classiferror');
    lossset(end+1) = lossAdd;
    % update fmin
    fmin = min(lossset);
    % update model
    gprMdl = fitrgp(paraset,lossset);
    
    
    iter=iter+1;
    

end
    
        
figure()
scatter(paraset,lossset)
title('Matern 5/2 kernel hyperparameter optimization')
xlabel('Scale')
ylabel('Loss')

[M,I] = min(lossset);
para = paraset(I,:);




end