function drawconfun3(xplot,SVMModels,PSVMModels,numSubset,SVMModel,PSVMModel)

K = size(xplot,1);
F = zeros(K,1);
Var = zeros(K,1);

for i=1:K
    x = xplot(i,:);
% committee predictions
%     labelj = zeros(numSubset,1);
%     scorej = zeros(numSubset,2);
    postprobj = zeros(numSubset,2);
    for j = 1:numSubset
%         SVMModelj = SVMModels{j};
%         [labelj(j, 1),scorej(j,:)] = predict(SVMModelj,x);
       PSVMModelj = PSVMModels{j};
       [~,postprobj(j,:)] = predict(PSVMModelj,x);
    end
   

% variance of label
% 	yVar = var(labelj(:, 1));
% 	sCon = sqrt(yVar);
% variance of score
%     yVar = var(scorej(:,2));
%     sCon = sqrt(yVar);
% variance of probability
    yVar = var(postprobj(:,2));
    sCon = sqrt(yVar);
    

% main model prediction
%     [label,score] = predict(SVMModel,x);
    [~,postprob] = predict(PSVMModel,x);
    

    % EI_feas = sCon*normpdf((0-ypredCon)/sCon)   % original EI_feas function
     % EI_feas = sCon*normpdf((-1-label)/sCon);
    % EI_feas = sCon*normpdf((0-score(2))/sCon);
     EI_feas = sCon*normpdf((0.5-postprob(2))/sCon);   

    Var(i) = yVar;
    F(i) = EI_feas;
    
end

Var_surf = reshape(Var,100,100);
F_surf = reshape(F, 100, 100);
x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);

figure()
contour(x1Plot_surf,x2Plot_surf,F_surf)
title('EI_{feas}')
colorbar

figure()
contour(x1Plot_surf,x2Plot_surf,Var_surf)
title('Variance')
colorbar
end