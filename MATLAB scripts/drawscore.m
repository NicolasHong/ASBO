function drawscore(xplot,SVMModel)
K = size(xplot,1);
score = zeros(K,2);

for i=1:K
    x = xplot(i,:);
    [~,score(i,:)] = predict(SVMModel,x);
end

score_surf = reshape(score(:,2), 100, 100);
x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);

figure()
contour(x1Plot_surf,x2Plot_surf,score_surf)
colorbar
title('score')

figure()
surf(x1Plot_surf,x2Plot_surf,score_surf)
colorbar
title('score_{surf}')