function neardist = caldX(X)
% X = [1 1;
%     2 1;
%     2 3;
%     1 1.1];

n = size(X,1);
dist = zeros(n);
for i = 1:n
    for j = 1:n        
        dist(i,j) = norm(X(i,:) - X(j,:));        
    end
end
maxdist = max(dist,[],'all');
dist = dist + eye(n)*maxdist;

neardist = min(dist,[],'all');