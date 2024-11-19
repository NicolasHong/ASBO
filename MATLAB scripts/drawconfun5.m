function drawconfun5(xplot,X,SVMModel)

K = size(xplot,1);
N = size(X,1);
dist_min = zeros(K,1);
scoreabs = zeros(K,1);
F = zeros(K,1);

for i = 1:K
    dist = zeros(N,1);
    for j = 1:N
        dist(j) = norm(xplot(i,:) - X(j,:));
    end
    dist_min(i) = min(dist);
    [~,score] = predict(SVMModel,xplot(i,:));
    scoreabs(i) = abs(score(1));
    F(i) = dist_min(i) - scoreabs(i);
end

dist_min_surf = reshape(dist_min, 100, 100);
score_surf = reshape(scoreabs, 100, 100);
F_surf = reshape(F, 100, 100);
x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);

figure()
contour(x1Plot_surf,x2Plot_surf,dist_min_surf)
title('dist min')
colorbar

figure()
contour(x1Plot_surf,x2Plot_surf,score_surf)
title('score')
colorbar

figure()
contour(x1Plot_surf,x2Plot_surf,F_surf)
title('F')
colorbar