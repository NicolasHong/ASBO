clear; close all; clc;
rng default  % For reproducibility

testCon = @test2DKG_ex3_con;
testObj = @test2DKG_ex3_obj;
numStr = 'example 3';
lb = [0, 0];
ub = [1, 1];
xGlob = [0.4, 0.3];
yGlob = 0;
d = 2;
numLoc = 49;

% lhs sampling
k = 20;
Xlhs = lhsdesign(k, d);   % design points
X = zeros(k, d);
for i = 1:k
    X(i,:) = lb + (ub - lb).*Xlhs(i,:);
end

Ycon = zeros(k, 1);
Yobj = zeros(k, 1);
for i = 1:k
    Ycon(i) = testCon(X(i, :));
    Yobj(i, :) = testObj(X(i, :));
end
Ycon(Ycon<=0)=-1;   %feasible
Ycon(Ycon>0)=1;    %infeasible
XiniFeasSet = [];
XiniInfeasSet = [];
for i=1:k
    if Ycon(i)<=0
        XiniFeasSet(end+1, :) = X(i,:);
    else
        XiniInfeasSet(end+1, :) = X(i,:);
    end
end


box = 10;        % try[10~1000]
sigma = 0.2; 
SVMModel = fitcsvm(X,Ycon,'Standardize',true,...
         'KernelFunction','rbf','BoxConstraint',box,...
         'KernelScale',sigma);
  
     
xplot = gridsamp([lb; ub], 100);
K = size(xplot, 1);
for ii = 1:K 
    [yPredcon(ii, :),Score(ii,:)] = predict(SVMModel,xplot(ii, :));      
 %   [yPredobj(ii, :), ~,~] = predictor(xplot(ii, :), modelobj);
end 
x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);
yPred_surf0con = reshape(Score(:,2), 100, 100);
%yPred_surf0obj = reshape(yPredobj, 100, 100);

figure()
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
set(HH1, 'LineWidth', 1.5, 'LineColor', 'blue', 'LineStyle', '-');
% contour(x1Plot_surf, x2Plot_surf, yPred_surf0obj, 'ShowText', 'on')
scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),'blue','fill')
scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),'red','fill')
hold off
legend('SVM for feas fun','Samples')
title([numStr, ' - box=',num2str(box),', scale=',num2str(sigma)])
xlabel('x1');
ylabel('x2'); 

