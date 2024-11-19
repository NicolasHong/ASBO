function [c,ceq] = confun1(x,SVMModel,ep)

ceq = [];
ScoreSVMModel = fitPosterior(SVMModel);
[~,postprob] = predict(ScoreSVMModel,x);
c = abs(postprob(1)-0.5) - ep;
end

