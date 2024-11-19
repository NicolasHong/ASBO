function drawvariance(K,numSubset,SVMModels,xplot)

for i = 1:K
    
    for j = 1:numSubset
        SVMModelj = SVMModels{j};
        [yPredcon(j, 1),yPredscore(j,:)] = predict(SVMModelj,xplot(i, :));
    end
    
    yVar(i,:) = var(yPredcon(:, 1));
    yVarscore(i,:) = var(yPredscore(:,2));
end

x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);

yVar_surf = reshape(yVar, 100, 100);
yVarscore_surf = reshape(yVarscore, 100, 100);

figure()
contour(x1Plot_surf, x2Plot_surf, yVar_surf)
title('varriance of label')

figure()
contour(x1Plot_surf, x2Plot_surf, yVarscore_surf)
title('variance of score')
colorbar