function [SVMModels,PSVMModels,figuresvmcmt] = kfoldsvmplot...
    (X,Ycon,numSubset,xplot,numGrid,hyperpts,cm)

k=size(X,1);
c = cvpartition(k,'KFold',numSubset);
SVMModels = cell(numSubset,1);
PSVMModels = cell(numSubset,1);

% figuresvmcmt = figure;
for i = 1:numSubset
    set = training(c,i);
    Xset = X(set,:);
    Yconset = Ycon(set);
    Xset_feas = Xset(Yconset<0,:);
    Xset_infeas = Xset(Yconset>=0,:);
    hyperparaopt = selectRbfPara(numGrid,hyperpts,Xset,Yconset,cm);
    SVMModeli = fitcsvm(Xset,Yconset,'Standardize',true,'KernelFunction','rbf',...
        'Cost',cm,'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));     
      
    SVMModels{i}=SVMModeli;
    PSVMModeli = fitPosterior(SVMModeli); 
    PSVMModels{i} = PSVMModeli;
    
%     subplot(2,5,i)
%     K = size(xplot,1);
% %     label = zeros(K,1);
% %     score = zeros(K,2);
%     postprob = zeros(K,2);
%     for j=1:K
%         x = xplot(j,:);
% %         [label(j),score(j,:)] = predict(SVMModeli,x);
%          [~,postprob(j,:)] = predict(PSVMModeli,x);
%     end
% %    label_surf = reshape(label, 100, 100);
% %     score_surf = reshape(score(:,2), 100, 100);
%      prob_surf = reshape(postprob(:,2), 100, 100);
%     x1Plot_surf = reshape(xplot(:, 1), 100, 100);
%     x2Plot_surf = reshape(xplot(:, 2), 100, 100);
%     
% %    contour(x1Plot_surf,x2Plot_surf,label_surf)
% %      contour(x1Plot_surf,x2Plot_surf,score_surf)
%      contour(x1Plot_surf,x2Plot_surf,prob_surf)
%     
%     hold on
%     if ~isempty(Xset_feas)
%         scatter(Xset_feas(:,1),Xset_feas(:,2),'b*')
%     end
%     if ~isempty(Xset_infeas)
%     scatter(Xset_infeas(:,1),Xset_infeas(:,2),'r*')
%     end
%     colorbar
%     title('score')
    
    
end