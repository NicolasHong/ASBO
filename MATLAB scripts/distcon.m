function [c,ceq] = distcon(x,X,thr)

N = size(X,1);
dist = zeros(N,1);
for i = 1:N
    dist(i) = norm(x - X(i,:));
end
dist_min = min(dist);

% c needd to be <0 for fmincon
c = thr - dist_min;   

ceq = [];