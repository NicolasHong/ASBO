function y = objfun1(x,X)
% find fartheset point x to existing points X
for i = 1:size(X,1)
    dist(i) = norm(x - X(i,:));

end

neardist = min(dist);
y = -neardist;