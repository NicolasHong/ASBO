function [s,d] = confun10(x,X,SVMModel,d_scale)
%Reference - Pan, Q., & Dias, D. (2017). An efficient reliability method combining adaptive Support Vector Machine and Monte Carlo Simulation. Structural Safety, 67, 85-95. doi:10.1016/j.strusafe.2017.04.006

N = size(X,1);
dist = zeros(N,1);
for i = 1:N
    dist(i) = norm(x - X(i,:));
end
d = min(dist)/d_scale;


[~,score] = predict(SVMModel,x);
s = abs(score(1));


end