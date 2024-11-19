function f = confun4(x,X,SVMModel)
% dist_min as uncertainty

N = size(X,1);
dist = zeros(N,1);
for i = 1:N
    dist(i) = norm(x - X(i,:));
end
dist_min = min(dist);

ScoreSVMModel = fitPosterior(SVMModel);
[~,postprob] = predict(ScoreSVMModel,x);

%EI_feas = sCon*normpdf((0-ypredCon)/sCon)   
EI_feas = dist_min*normpdf((0.5-postprob(1))/dist_min);   

f = -EI_feas;
end