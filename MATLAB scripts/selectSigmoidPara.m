function para = selectSigmoidPara(X,Ycon)

d = 2;
lb = [1e-5, 1e-5];
ub = [1e3, 1e3];

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
    passkSigmoidPara(value);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'CVPartition',c,'KernelFunction','mysigmoid');
    lossset(i) = kfoldLoss(SVMModel);
end

gprMdl = fitrgp(paraset,lossset);


%% bayesian optimization for hyperparameters
iter = 1;
while iter < 10^d
    
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
    passkSigmoidPara(value);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'CVPartition',c,'KernelFunction','mysigmoid');
    lossAdd = kfoldLoss(SVMModel);
    lossset(end+1) = lossAdd;
    % update fmin
    fmin = min(lossset);
    % update model
    gprMdl = fitrgp(paraset,lossset);
    
    
    iter=iter+1;
    

end
    
        
figure()
scatter3(paraset(:,1),paraset(:,2),lossset)
title('Sigmoid kernel hyperparameter optimization')
xlabel('Gamma')
ylabel('C')
zlabel('Loss')

[M,I] = min(lossset);
para = paraset(I,:);




end
    
