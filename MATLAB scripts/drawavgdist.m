function drawavgdist(xplot,X)

k = size(X,1);
K = size(xplot,1);

for i = 1:K
    for j = 1:k
        dist(j,:) = norm(xplot(i, :) - X(j,:));
    end
    dist_avg(i,:) = sum(dist)/k;
end

dist_avg_surf = reshape(dist_avg, 100, 100);
x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);

figure()
contour(x1Plot_surf, x2Plot_surf, dist_avg_surf)
title('dist_avg')