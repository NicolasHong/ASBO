function f = confun2(x,X,SVMModel)
% dist_avg as uncertainty

N = size(X,1);
dist = zeros(N,1);
for i = 1:N
    dist(i) = norm(x - X(i,:));
end
dist_avg = sum(dist)/N;

ScoreSVMModel = fitPosterior(SVMModel);
[~,postprob] = predict(ScoreSVMModel,x);

%EI_feas = sCon*normpdf((0-ypredCon)/sCon)   
EI_feas = dist_avg*normpdf((0.5-postprob(1))/dist_avg);   

f = -EI_feas;
end