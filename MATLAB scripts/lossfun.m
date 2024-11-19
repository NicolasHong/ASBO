function lossval = lossfun(x,X,Ycon)
% Define a custom loss function
    
    % Assign globals the values passed from bayesopt
    global kMatern52l;
    kMatern52l = x.M52Scale;
    C = x.M52BoxConstraint;

    c = cvpartition(size(X,1),'KFold',10);
    % Call the loss funciton with the fitcsvm function
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'BoxConstraint',C,...
        'CVPartition',c,'KernelFunction','mymatern52');
    lossval = kfoldLoss(SVMModel,'LossFun','logit');
    
    
end