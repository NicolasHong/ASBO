function hyperparaopt = selectRbfPara(numGrid,hyperpts,X,Ycon)

numGridAll = numGrid^2;
lossVal = zeros(numGridAll,1);
if std(Ycon)==0
    hyperparaopt = [1 1];
else
    for i = 1:numGridAll
        sigma = hyperpts(i,1);
        box = hyperpts(i,2);
    %     lossVal(i) = kfoldLoss(fitcsvm(X,Ycon,'Standardize',true,'CVPartition',c,...
    %         'KernelFunction','rbf','BoxConstraint',box,...
    %         'KernelScale',sigma),'LossFun','classiferror');
        lossVal(i) = loss(fitcsvm(X,Ycon,'Standardize',true,...
         'KernelFunction','rbf','BoxConstraint',box,...
         'KernelScale',sigma),X,Ycon); 
    end
    [M,~] = min(lossVal);
    if M>0
        error('Min of loss value > 0. Misclassification exists.')
    end
    hyperptsmin = hyperpts(lossVal==M,:);
    [~,I] = max(hyperptsmin(:,1));   %max scale
    hyperparaopt = hyperptsmin(I,:);
end


% plot
% scale_surf = reshape(hyperpts(:, 1), numGrid, numGrid);
% box_surf = reshape(hyperpts(:, 2), numGrid, numGrid);
% loss_surf = reshape(lossVal, numGrid, numGrid);
% figure()
% h=gca;
% surf(scale_surf,box_surf,loss_surf)
% xlabel('KernelScale');
% ylabel('BoxCon');
% zlabel('LossFunction');
% set(h,'xscale','log');
% set(h,'yscale','log');
% title('Hyperparameter space')