function f = disthravgmin(X)

D = pdist(X);
Z = squareform(D);
Z(Z==0) = nan;

for i=1:size(X,1)
    dist_min(i) = min(Z(i,:));
end

D_avg = mean(dist_min);
f = 0.5*D_avg;