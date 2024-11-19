clear; close all; clc;
% rng default  % For reproducibility


testCon = @test2DKG_ex3_con;
testObj = @test2DKG_ex3_obj;
numStr = 'example 3';
lb = [0, 0];
ub = [1, 1];
xGlob = [0.4, 0.3];
yGlob = 0;
d = 2;
numLoc = 49;

% testObj = @test2DKG_gomez_obj;
% testCon = @test2DKG_gomez_con;
% numStr = 'gomez';
% lb = [-1, -1];
% ub = [1, 1];
% xGlob = [0.1093, -0.6234];
% yGlob = -0.9711;
% d = 2;
% numLoc = 49;



% lhs sampling
k = 20;
Xlhs = lhsdesign(k, d); % design points
X = zeros(k, d);
for i = 1:k
    X(i,:) = lb + (ub - lb).*Xlhs(i,:);
end
% X=[X;
%     [0.5 0.5];
%     [0.4 0.4];
%     [0.45 0.524];
%     [0.55 0.49];];
% k=24;
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



GlobalFig = figure();

%% distance to nearest neighbor
% for iter = 1:50
%     Xtest = generateXtest(d,ub,lb,2000);
%     parfor i=1:size(Xtest,1)
%         x = Xtest(i,:);
%         fval(i) = -cald2(x,X);
%     end
%     [M,I] = min(fval);
%     Xtestopt = Xtest(I,:);
% 
%     X = [X; Xtestopt];
%     plotglobal(X,GlobalFig,iter)
% end


%% 1-r*pinv(R)*r
for iter = 1:50
    cm = [0,1;1,0];
    numGrid = 10;
    hyperpts = gridsamplog([-3 -3; 3 3], numGrid); %range=[1e-3,1e3] for both scale and boxcon
    hyperparaopt = selectRbfPara(numGrid,hyperpts,X,Ycon);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'KernelFunction','rbf',...
      'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));
    Xtest = generateXtest(d,ub,lb,2000);
    parfor i=1:size(Xtest,1)
        x = Xtest(i,:);
        fval(i) = objfun3(x,X,hyperparaopt);
    end
    [M,I] = min(fval);
    Xtestopt = Xtest(I,:);
    X = [X; Xtestopt];
    Yconadd = testCon(Xtestopt);
    Ycon = [Ycon; Yconadd];
    Ycon(Ycon<=0)=-1;   %feasible
    Ycon(Ycon>0)=1;    %infeasible
    plotglobal(X,GlobalFig,iter)
    
end
 

