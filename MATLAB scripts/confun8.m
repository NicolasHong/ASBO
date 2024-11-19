function [f,p,s,pdf] = confun8(x,SVMModels,PSVMModels,numSubset,SVMModel,PSVMModel)

% committee predictions
labelj = zeros(numSubset,1);
scorej = zeros(numSubset,2);
postprobj = zeros(numSubset,2);
for j = 1:numSubset
   SVMModelj = SVMModels{j};
   [labelj(j, 1),scorej(j,:)] = predict(SVMModelj,x);
   PSVMModelj = PSVMModels{j};
   [~,postprobj(j,:)] = predict(PSVMModelj,x);
end
   

% variance of label
% 	yVar = var(labelj(:, 1));
% 	sCon = sqrt(yVar);       
% variance of score
%     yVar = var(scorej(:,2));
%     sCon = sqrt(yVar);
% variance of probability
     yVar = var(postprobj(:,2));
     sCon = sqrt(yVar);
    

% main model prediction
    [label,score] = predict(SVMModel,x);
    [~,postprob] = predict(PSVMModel,x);
    

% EI_feas = sCon*normpdf((0-ypredCon)/sCon)   % original EI_feas function
%   EI_feas = sCon*normpdf((-1-label)/sCon);
%   EI_feas = sCon*normpdf((0-score(2))/sCon);
if sCon == 0
    EI_feas = 0;
else
    EI_feas = sCon*normpdf((0.5-postprob(2))/sCon);   
end

f = EI_feas;
p = 1-postprob(2);
s = sCon;
pdf = normpdf((0.5-postprob(2))/sCon);
end