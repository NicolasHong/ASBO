% with 4 stages - local optimization stage added
% feasibility stage does not use EIfeas


clear; close all; clc;
rng (3)  % For reproducibility
fffthr = 0.05;   % threshold value to measure svm model changes
objthr = 0.02;   % threshold value to measure fmin_diff
% numAdpptsFeas = 1;
% numAdppts = 1;

numRep = 1;
warning off

%% test problem
% testObj = @test2DKG_braninC_obj;
% testCon = @test2DKG_braninC_con;
% numStr = 'braninC';
% lb = [-5, 0];
% ub = [10, 15];
% xGlob = [9.42478, 2.475; pi, 2.275];
% yGlob = 0.3979;
% d = 2;
% numLoc = 49;

% testObj = @test2DKG_newBranin_obj;
% testCon = @test2DKG_newBranin_con;
% numStr = 'newBranin';
% lb = [-5, 0];
% ub = [10, 15];
% xGlob = [3.273, 0.0489];
% yGlob = -268.7879;
% numLoc = 49;
% d = 2;

% testObj = @test2DKG_camelback_obj;
% testCon = @test2DKG_camelback_con;
% numStr = 'camelback';
% lb = [-3, -2]; 
% ub = [3, 2];
% x0 = [1.776, -0.7755];
% xGlob = [1.8363, -0.8019];
% yGlob = -1.8363;
% d = 2;
% numLoc = 49;

testCon = @test2DKG_ex3_con;
testObj = @test2DKG_ex3_obj;
numStr = 'example 3';
lb = [0, 0];
ub = [1, 1];
xGlob = [0.4, 0.3];
yGlob = 0;
d = 2;
numLoc = 49;

% testCon = @test2DKG_sasena_con;
% numStr = 'sasenaCon';
% lb = [0, 0]; 
% ub = [1, 1];
% d = 2;
% numLoc = 49;

% testObj = @test2DKG_gomez_obj;
% testCon = @test2DKG_gomez_con;
% numStr = 'gomez';
% lb = [-1, -1];
% ub = [1, 1];
% xGlob = [0.1093, -0.6234];
% yGlob = -0.9711;
% d = 2;
% numLoc = 49;

% testCon = @test2DKG_gomez_con;
% testObj = @test_michal_2;
% numStr = 'test problem';
% lb = [-2, -2];
% ub = [2, 2];
% xGlob = [0.1, -0.5]; %?
% yGlob = -1.8013;  %?
% d = 2;
% numLoc = 49;


% testObj = @test3DKG_qcp4_obj;
% testCon = @test3DKG_qcp4_con;
% numStr = 'qcp4';
% lb = [0, 0, 0];
% ub = [2, 3, 3];
% xGlob = [0.5, 0, 3];
% yGlob = -4;
% d = 3;
% numLoc = 30;

% testObj = @PrG4f;
% testCon = @PrG4c;
% numStr = 'g4';
% lb = [78, 33, 27, 27, 27];
% ub = [102, 45, 45, 45, 45];
% xGlob = [78,33,29.995,45,36.7758];
% yGlob = -30665.539;
% d = 5;
% numLoc = 50;



%% plot settings and plot original functions
xplot = gridsamp([lb; ub], 100);
K = size(xplot, 1);

parfor i = 1:K
    yOrigcon(i, :) = testCon(xplot(i, :)); 
    yOrigobj(i, :) = testObj(xplot(i, :));
end

xfeasSet = [];
for i = 1:K
    if yOrigcon(i)<=0
        xfeasSet(end+1, :) = xplot(i,:);
    end
end

x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);
yOrig_surf0con = reshape(yOrigcon, 100, 100);
yOrig_surf0obj = reshape(yOrigobj, 100, 100);

figure()
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yOrig_surf0con, [0, 0]);
set(HH1, 'LineWidth', 1.5, 'LineColor', 'blue', 'LineStyle', '-');
plot(xfeasSet(:,1),xfeasSet(:,2),'.','MarkerEdgeColor','b')
contour(x1Plot_surf, x2Plot_surf, yOrig_surf0obj, 'ShowText', 'on')
scatter(xGlob(:,1),xGlob(:,2),'fill')
hold off
title(numStr)
xlabel('x1');
ylabel('x2');
colorbar 


% mae = zeros(numRep,1);
stage_store_cell = cell(numRep,1);
for rep=1:numRep
% for rep = 1:10
%     rng (rep-1);
%% generate initial points

% load data
% load result_3.mat X  % ex3 problem from Zilong's rbf
% load result_3.mat y
% load result_4.mat X  % branincon problem from Zilong's rbf
% load result_4.mat y
% Ycon = y;
% k = size(X,1);
% numLoc = k;

 % lhs sampling
k = numLoc;
Xlhs = lhsdesign(k, d); % design points
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

% categorize all sample points
XiniFeasSet = [];
XiniInfeasSet = [];
for i=1:numLoc
    if Ycon(i)<=0
        XiniFeasSet(end+1, :) = X(i,:);
    else
        XiniInfeasSet(end+1, :) = X(i,:);
    end
end
XFeasSet = XiniFeasSet;
XInfeasSet = XiniInfeasSet;



%% train model
% Feasibility function
% RBF kernel bayesopt - check main_ConOpt_svm_20221027

% RBF kernel - grid sampling and no overfitting
cm = [0,1;1,0];
numGrid = 10;
hyperpts = gridsamplog([-3 -3; 3 3], numGrid); %range=[1e-3,1e3] for both scale and boxcon
hyperparaopt = selectRbfPara(numGrid,hyperpts,X,Ycon);
SVMModel = fitcsvm(X,Ycon,'Standardize',true,'KernelFunction','rbf',...
    'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));

% % Matern 5/2 kernel
% global kMatern52l;
% % Declare Optimizable Variables
% varA = optimizableVariable('M52Scale', [1e-3 1e3],'Transform','log');
% varB = optimizableVariable('M52BoxConstraint', [1e-3 1e3],'Transform','log');
% % Create Function Wrapper
% lossfun2 = @(x)lossfun(x,X,Ycon);
% % Run Bayes Optimization
% results = bayesopt(lossfun2, [varA varB],'IsObjectiveDeterministic',true,...
%     'AcquisitionFunctionName','expected-improvement-plus',...
%     'ExplorationRatio',0.5,...
%     'MaxObjectiveEvaluations',200,...
%     'UseParallel',false,...
%     'NumSeedPoints',4);
% z = bestPoint(results);
% hyperparaopt = table2array(z);
% kMatern52l = hyperparaopt(1);
% boxcon = hyperparaopt(2);
% SVMModel = fitcsvm(X,Ycon,'Standardize',true,'BoxConstraint',boxcon,...
%     'KernelFunction','mymatern52');

% feasibility model outputs
SV_standardized = SVMModel.SupportVectors;
SV = [];
SV(:,1) = SV_standardized(:,1)*SVMModel.Sigma(1) + SVMModel.Mu(1);
SV(:,2) = SV_standardized(:,2)*SVMModel.Sigma(2) + SVMModel.Mu(2);
%SVL = SVMModel.SupportVectorLabels;


% objective function
[ModeltypeObj,thetaObj] = modelselectcon(Yobj,X);
modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);



%% plot initial models
for ii = 1:K 
    [yPredcon(ii, :),Score(ii,:)] = predict(SVMModel,xplot(ii, :));      
    [yPredobj(ii, :), ~,~] = predictor(xplot(ii, :), modelobj);
end   
yPred_surf0con = reshape(Score(:,2), 100, 100);
yPred_surf0obj = reshape(yPredobj, 100, 100);


figure()
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
set(HH1, 'LineWidth', 1.5, 'LineColor', 'blue', 'LineStyle', '-');
contour(x1Plot_surf, x2Plot_surf, yPred_surf0obj, 'ShowText', 'on')
scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),'b.')
scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),'r.')
scatter(SV(:,1),SV(:,2),'ks')
hold off
legend('SVM for feas fun','GP for obj fun','Feasible points',...
    'Infeasible points','Support vectors')
title([numStr, ' - initial surrogates'])
xlabel('x1');
ylabel('x2'); 

% Confusion chart
% for i = 1:size(X,1) 
%     [yPredconX(i, :),~] = predict(SVMModel,X(i, :));
% end
% figure()
% ConfusionTest = confusionchart(Ycon,yPredconX);  
% title('Confusion chart of initial SVM')
    
% Calculate feasibility metrics
% [f] = calcCorr(lb, ub, testCon, SVMModel,100);
% disp(['CF% = ',num2str(f(1)*100),...
%     '%,  CIF% = ',num2str(f(2)*100),...
%     '%,  NC% = ',num2str(f(3)*100),'%'])
       
% Calculate probability contour - Basic sigmoid
% [SVMModelP,ScoreParameters] = fitPosterior(SVMModel); 
% for ii = 1:K
%     [~,PostProbs(ii, :)] = predict(SVMModelP,xplot(ii, :));     
% end
% yPP_surf0obj = reshape(PostProbs(:,1), 100, 100);

% Calculate probability contour - Modified sigmoid
[A, B] = myfitPosterior(X,Ycon,XFeasSet,XInfeasSet,SVMModel); 
for ii = 1:K
    x = xplot(ii,:);
    dpos = cald(x,XiniInfeasSet);
    dneg = cald(x,XiniFeasSet);
%    Bm = dneg/(dpos+1e-10) - dpos/(dneg+1e-10);
    Bm=1;
    PostProbsCal(ii,:) = 1./(1+exp(A.*Score(ii,2)+B*Bm));
    PostProbsMod = 1-PostProbsCal;
end
yPPcal_surf0obj = reshape(PostProbsMod(:,1), 100, 100);

figure()
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
set(HH1, 'LineWidth', 2, 'LineColor', 'blue', 'LineStyle', '-');
% [CC2, HH2] = contour(x1Plot_surf, x2Plot_surf, yPP_surf0obj,[0.95 0.5 0.05]);
% set(HH2, 'LineWidth', 1, 'LineColor', 'k', 'LineStyle', '--','ShowText', 'on'); 
[CC3, HH3] = contour(x1Plot_surf, x2Plot_surf, yPPcal_surf0obj);
set(HH3, 'LineWidth', 1,'LineStyle', '--','ShowText', 'on'); 
scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),'b.')
scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),'r.')
scatter(SV(:,1),SV(:,2),'ks')
hold off
legend('SVM boundary','P(-1|x)',...
    'Feasible points','Infeasible points','Support vectors')
title([numStr, ' - probability contour'])
xlabel('x1');
ylabel('x2'); 
colorbar


%% Adaptive sampling initialization
iter=0;
XadpFeasSet = [];
XadpInfeasSet = [];
stage = 0;
stage_store = [];
A_store = [A];
B_store = [B];
fmin_store = [];
Yobj_feas = Yobj(Ycon<=0);
fmin = min(Yobj_feas);
fmin_store(end+1) = fmin;
thr = disthrcst(ub,lb);
fracfeas_store = [];
fracfeas_diff_store = [];
fracfeas = computeFeasFrac(SVMModel,xplot);
fracfeas_store(end+1) = fracfeas; 
options = optimoptions('fmincon','Display','iter','OptimalityTolerance',1e-16);

ResultFig = figure('position', [200   200   1400   500]);
subplot(1,2,2)
hold on
scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),30,'bo','filled')
scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),30,'ro','filled')
scatter(xGlob(:,1),xGlob(:,2),100,'pentagram','filled')
hold off
xlabel('x1');
ylabel('x2');
colorbar 


if yGlob==0
    fmin_diff = abs((fmin-yGlob)/1e-4);
else
    fmin_diff = abs((fmin-yGlob)/yGlob);
end

fffid=1;


%% Adaptive sampling
tic
while fmin_diff > objthr   % stopping criterion for entire algorithm
% while iter<numAdppts
    
    iter = iter + 1;
    SVid = SVMModel.IsSupportVector;
    SVidX = SVid(end);
    numSV = sum(SVid);
    fracSV = numSV/size(X,1)

% decide which stage to go
    if stage == 0 % && fracSV<0.5  % start
        stage = 1;  % feasibility 
%     elseif stage ==0 && fracSV>=0.5
%         stage = 4;
    elseif samex ==1
        stage = 4;
    elseif stage == 1 && numStage1(stage_store)>10 && fffid==0
        stage = 2;  % constrained optimization        
    %elseif stage == 2 && length(stage_store(stage_store==2))>10 && (cald2(X(end,:),X)<thr || std(fmin_store(end-9:end))<1e-6)
    elseif (numStage2(stage_store)>10 && std(fmin_store(end-4:end))<1e-6)
        stage = 3; % local optimization    
%     elseif numStage3(stage_store)>2 && Ycon(end)>0 && fffthr>0.025 % local optimization is near the boundary
%         stage = 1; % go back to feasibility
%         fffthr = fffthr/2;
    elseif numStage3(stage_store)>2 % enough local opt
        stage = 4; % global exploration    
    elseif stage==4 && Ycon(end)<=0 && Yobj(end)<mean(Yobj) && SVidX
    %elseif stage==5 && Ycon()<=0 && Yobj(end)<fmin  
        stage = 1;
        fffthr=0.05;
    end

    
% do adaptive sampling
    if stage == 1
        disp(['iter = ',num2str(iter),', stage = feasibility']);
        A = A_store(end);
        B = B_store(end);
        d_scale = caldmaxX(X);
        for i=1:size(X,1)
            [~,ScoreX(i,:)] = predict(SVMModel,X(i,:));
        end
        score_scale = max(ScoreX(:,1));
        Xtest = generateXtest(d,ub,lb,2000);
        parfor ii=1:size(Xtest,1)
            x=Xtest(ii, :);
            L(ii) = confun9(x,X,A,B,SVMModel,d_scale,score_scale);
        end
        [M,I] = min(L);
        Xtestopt = Xtest(I,:);
        [Xf,fval] = fmincon(@(x)confun9(x,X,A,B,SVMModel,d_scale,score_scale),Xtestopt,[],[],[],[],lb,ub,@(x)distcon(x,X,thr),options);
        if mod(iter,10)==0
            parfor ii=1:size(xplot,1)
                x=xplot(ii, :);
                [f(ii),Pwse(ii),dist_min(ii)]=confun5(x,X,A,B,SVMModel,d_scale,score_scale);
                [~,Score(ii,:)] = predict(SVMModel,x);
            end
            figure()
            f_surf = reshape(f,[sqrt(K),sqrt(K)]);
            yPred_surf0con = reshape(Score(:,2), 100, 100);
            contour(x1Plot_surf,x2Plot_surf, f_surf)
            hold on
            [Ccon, Hcon] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
            set(Hcon, 'LineWidth', 1, 'LineColor', 'blue', 'LineStyle', '--');
            plot(Xf(:,1),Xf(:,2),'r*')
            plot(X(:,1),X(:,2),'k.')
            title(['L stage1 - iter=',num2str(iter)])
            colorbar
%             figure()
%             Pwse_surf = reshape(Pwse,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, Pwse_surf)
%             title(['Probability of wrong classification - iter=',num2str(iter)])
%             colorbar
%             figure()
%             dist_min_surf = reshape(dist_min,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, dist_min_surf)
%             title(['Distance to nearest neighbor - iter=',num2str(iter)])
%             colorbar
        end

%     if stage == 1
%         disp(['iter = ',num2str(iter),', stage = feasibility']);
%         PSVMModel = fitPosterior(SVMModel);  % temporary
%         numSubset = 10;
%         [SVMModels,PSVMModels] = kfoldsvm(X,Ycon,numSubset,numGrid,hyperpts,cm);
%         Xtest = generateXtest(d,ub,lb,2000);
%         parfor ii=1:size(Xtest,1)
%             x=Xtest(ii, :);
%             [EIfeasha(ii),~,~,~]=...
%                 confun8(x,SVMModels,PSVMModels,numSubset,SVMModel,PSVMModel);
%         end
%         [M,I] = max(EIfeasha);  
%         Xtestopt = Xtest(I,:);
%         [Xf,fval] = fmincon(@(x)confun3(x,SVMModels,PSVMModels,numSubset,SVMModel,PSVMModel),Xtestopt,[],[],[],[],lb,ub,@(x)distcon(x,X,thr),options);
%         if mod(iter,10)==0
%             parfor ii=1:size(xplot,1)
%             x=xplot(ii, :);
%             [EIFeas(ii),probFeas(ii),sConFeas(ii),pdfFeas(ii)]=...
%                 confun8(x,SVMModels,PSVMModels,numSubset,SVMModel,PSVMModel);
%             end
%             figure()
%             EIfeas_surf = reshape(EIFeas,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, EIfeas_surf)
%             hold on
%             plot(Xf(:,1),Xf(:,2),'r*')
%             title(['EIfeas - iter=',num2str(iter)])
%             colorbar
%             figure()
%             sConFeas_surf = reshape(sConFeas,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, sConFeas_surf)
%             title(['Standard deviation of probablity of feasibility - iter=',num2str(iter)])
%             colorbar
%             figure()
%             probFeas_surf = reshape(probFeas,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, probFeas_surf)
%             title(['Probability of feasibility - iter=',num2str(iter)])
%             colorbar
%             figure()
%             pdfFeas_surf = reshape(pdfFeas,[sqrt(K),sqrt(K)]);
%             contour(x1Plot_surf,x2Plot_surf, pdfFeas_surf)
%             title(['Norm pdf in EIfeas - iter=',num2str(iter)])
%             colorbar
%         end
        
    elseif stage == 2  % constrained optimization
        disp(['iter = ',num2str(iter),', stage = constrained optimization'])
        A = A_store(end);
        B = B_store(end);
        Xtest = generateXtest(d,ub,lb,2000);
        parfor i=1:size(Xtest,1)
            x = Xtest(i,:);
            fval(i) = objfun2(x,modelobj,fmin);   %EI/LCB etc.
            cval(i) = confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel); %p(-1|x)>0.3
        end
        fval_feas = fval(cval<0);
        Xtest_feas = Xtest(cval<0,:);
        if isempty(Xtest_feas)
            [M,I] = min(fval);
            Xtestopt = Xtest(I,:);
        else
            [M,I] = min(fval_feas);
            Xtestopt = Xtest_feas(I,:);
        end
        [Xf,fval] = fmincon(@(x)objfun2(x,modelobj,fmin),Xtestopt,[],[],[],[],lb,ub,@(x)confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel),options);
        if confun7(Xf,A,B,XFeasSet,XInfeasSet,SVMModel)>0
            disp('stage 2 converge to infeasibe point')
            delta = (ub-lb)/10;
            lbcheck = Xtestopt-delta>=lb;
            lbdelta = (Xtestopt-delta).*lbcheck+lb.*~lbcheck;
            ubcheck = Xtestopt+delta<=ub;
            ubdelta = (Xtestopt+delta).*ubcheck+ub.*~ubcheck;
            Xtest = generateXtest(d,lbdelta,ubdelta,100);
            parfor i=1:size(Xtest,1)
                x = Xtest(i,:);
                fvall(i) = objfun2(x,modelobj,fmin);   %EI/LCB etc.
                cvall(i) = confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel); %p(-1|x)>0.3
            end
            fval_feas = fvall(cvall<0);
            Xtest_feas = Xtest(cvall<0,:);
            if isempty(Xtest_feas)
                [M,I] = min(fvall);
                Xtestopt = Xtest(I,:);
            else
                [M,I] = min(fval_feas);
                Xtestopt = Xtest_feas(I,:);
            end
            Xf = Xtestopt;
        end
        if mod(iter,10)==0
        parfor ii=1:size(xplot,1)
               x=xplot(ii, :);
               f(ii)=-objfun2(x,modelobj,fmin);
               [~,Score(ii,:)] = predict(SVMModel,x);
        end
        figure()
        f_surf = reshape(f,[sqrt(K),sqrt(K)]);
        yPred_surf0con = reshape(Score(:,2), 100, 100);
        contour(x1Plot_surf,x2Plot_surf, f_surf)
        hold on
        [Ccon, Hcon] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
        set(Hcon, 'LineWidth', 1, 'LineColor', 'blue', 'LineStyle', '--');
        plot(Xf(:,1),Xf(:,2),'r*')
        plot(X(:,1),X(:,2),'k.')
        title(['EI stage2 - iter=',num2str(iter)])
        colorbar
        end
    elseif stage == 3  %local optimization
        disp(['iter = ',num2str(iter),', stage = Local optimization'])
        delta = (ub-lb)/10;
        lbcheck = xmin-delta>=lb;
        lbdelta = (xmin-delta).*lbcheck+lb.*~lbcheck;
        ubcheck = xmin+delta<=ub;
        ubdelta = (xmin+delta).*ubcheck+ub.*~ubcheck;
        [Xf,fval] = fmincon(@(x)objfun4(x,modelobj),xmin,[],[],[],[],lbdelta,ubdelta,@(x)confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel),options);
        if confun7(Xf,A,B,XFeasSet,XInfeasSet,SVMModel)>0
            disp('stage 3 converge to infeasibe point')
            Xtest = generateXtest(d,lbdelta,ubdelta,100);
            parfor i=1:size(Xtest,1)
                x = Xtest(i,:);
                f3(i) = objfun4(x,modelobj);   
                c3(i) = confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel); %p(-1|x)>0.3
            end
            fval_feas = f3(c3<0);
            Xtest_feas = Xtest(c3<0,:);
            if isempty(Xtest_feas)
                [M,I] = min(f3);
                Xtestopt = Xtest(I,:);
            else
                [M,I] = min(fval_feas);
                Xtestopt = Xtest_feas(I,:);
            end
            Xf = Xtestopt;                     
        end
%         if mod(iter,1)==0
%             xplotdelta = gridsamp([lbdelta; ubdelta], 100);
%             x1Plotdelta_surf = reshape(xplotdelta(:, 1), 100, 100);
%             x2Plotdelta_surf = reshape(xplotdelta(:, 2), 100, 100);
%         parfor ii=1:size(xplot,1)
%                x=xplotdelta(ii, :);
%                f(ii)=-objfun4(x,modelobj);
%                [~,Score(ii,:)] = predict(SVMModel,x);
%         end
%         figure()
%         f_surf = reshape(f,[sqrt(K),sqrt(K)]);
%         yPred_surf0con = reshape(Score(:,2), 100, 100);
%         contour(x1Plotdelta_surf,x2Plotdelta_surf, f_surf)
%         hold on
%         [Ccon, Hcon] = contour(x1Plotdelta_surf, x2Plotdelta_surf, yPred_surf0con, [0, 0]);
%         set(Hcon,'LineWidth',1, 'LineColor','blue','LineStyle','--');
%         plot(Xf(:,1),Xf(:,2),'r*')
% %         plot(X(:,1),X(:,2),'k.')
%         title(['EI stage3 - iter=',num2str(iter)])
%         colorbar
%         end        
        
        
    elseif stage == 4  % global exploration
        disp(['iter = ',num2str(iter),', stage = global exploration'])
        Xtest = generateXtest(d,ub,lb,1000);
        parfor i=1:size(Xtest,1)
            x = Xtest(i,:);
            f4(i) = -cald2(x,X);
        end
        [M,I] = min(f4);
        Xtestopt = Xtest(I,:);
        %[Xf,fval] = fmincon(@(x)objfun3(x,X,hyperparaopt),Xtestopt,[],[],[],[],lb,ub,[],options);
        [Xf,fval] = fmincon(@(x)-cald2(x,X),Xtestopt,[],[],[],[],lb,ub,[],options);
 
    end

    
    
% check Xf
    Xa = [X;Xf];
    Xu = unique(Xa,'rows');
    samex=0;
    if size(Xa,1)>size(Xu,1)
        samex = samex+1;
        continue
       
    end

% update everything
    stage_store(end+1) = stage;

    [X,Ycon,Yobj,XadpFeasSet,XFeasSet,XadpInfeasSet,XInfeasSet,...
        SVMModel,A_store,B_store,modelobj,fmin_store,xmin] = update...
        (Xf,testCon,testObj,X,XadpFeasSet,XFeasSet,XadpInfeasSet,XInfeasSet,...
        Ycon,Yobj,numGrid,hyperpts,cm,A_store,B_store,fmin_store); 

    fracfeas = computeFeasFrac(SVMModel,xplot);
    fracfeas_store(end+1) = fracfeas; 
    fracfeas_diff = abs((fracfeas_store(end)-fracfeas_store(end-1))/fracfeas_store(end-1));
    fracfeas_diff_store(end+1) = fracfeas_diff;
    if length(fracfeas_diff_store)>=10
        fff = fracfeas_diff_store(end-9:end)<fffthr;
        if length(fff(fff==1))== 10
            fffid = 0;  % svm model not changing
        else
            fffid = 1;  % svm model still changing
        end
    end

    fmin = fmin_store(end);
    if yGlob==0
        fmin_diff = abs((fmin-yGlob)/1e-4);
    else
        fmin_diff = abs((fmin-yGlob)/yGlob);
    end

% plot
    plotresult(ResultFig,stage_store,SVMModel,Yobj,Ycon,XFeasSet,xplot,...
        modelobj,x1Plot_surf, x2Plot_surf, yOrig_surf0con,...
        XiniFeasSet,XiniInfeasSet,XadpFeasSet,XadpInfeasSet,...
        xGlob,iter,numStr)

end

toc




%% Results
% support vectors
SV_standardized = SVMModel.SupportVectors;
SV_final = [];
SV_final(:,1) = SV_standardized(:,1)*SVMModel.Sigma(1) + SVMModel.Mu(1);
SV_final(:,2) = SV_standardized(:,2)*SVMModel.Sigma(2) + SVMModel.Mu(2);
SVL = SVMModel.SupportVectorLabels;

Yobj_feas = Yobj(Ycon<=0);
[fmin,I] = min(Yobj_feas);
xmin = XFeasSet(I,:);

% plot results
    parfor i = 1:K
    [~,Score(i, :)] = predict(SVMModel,xplot(i, :));
    [yPredobj(i, :), ~,~] = predictor(xplot(i, :), modelobj);
    end

    yPred_surf0con = reshape(Score(:,2), 100, 100);
    yPred_surf0obj = reshape(yPredobj, 100, 100);
    
    figure('position', [693   217   861   708])
    hold on
    [CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yOrig_surf0con, [0, 0]);
    set(HH1, 'LineWidth', 1, 'LineColor', 'black', 'LineStyle', '-');
    [CC2, HH2] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
    set(HH2, 'LineWidth', 1, 'LineColor', 'blue', 'LineStyle', '--');
    contour(x1Plot_surf, x2Plot_surf, yPred_surf0obj, 'ShowText', 'on')
    scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),30,'bo','filled')
    scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),30,'ro','filled')
    scatter(XadpFeasSet(:,1),XadpFeasSet(:,2),'b*')   
    scatter(XadpInfeasSet(:,1),XadpInfeasSet(:,2),'r*')    
    scatter(SV_final(:,1),SV_final(:,2),70,'ks')
    scatter(xGlob(:,1),xGlob(:,2),100,'pentagram','filled')
    scatter(xmin(:,1),xmin(:,2),100,'ro')
    hold off
    legend('Feas fun','SVM for feas fun','GP for obj fun',...
           'Ini feas','Ini infeas','Adp feas','Adp infeas',...
           'Support vectors','True optimum','Optimal solution') 
    title(numStr)
    xlabel('x1');
    ylabel('x2');
    colorbar 

% plot fmin_store
figure()
plot([1:length(fmin_store)],fmin_store)
xlabel('iteration')
ylabel('fmin')
title('fmin')

% plot stage
figure()
plot([1:length(stage_store)],stage_store)
xlabel('Iteration')
ylabel('Stage')
title({'Stage index';'1=Feasibility stage, 2=Optimization stage,3=Local refinement stage,4=Global exploration stage'})
ylim([0 4])

% calculate metrics  
%     [f] = calcCorr(lb, ub, testCon, SVMModel,100);
%     disp(['CF% = ',num2str(f(1)*100),...
%     '%,  CIF% = ',num2str(f(2)*100),...
%     '%,  NC% = ',num2str(f(3)*100),'%'])

%     fmin_diff = abs((fmin-yGlob)/yGlob);


stage_store_cell{rep,1} = stage_store;    
mae(rep) = fmin_diff;    
end

maeavg = mean(mae)
maesd = std(mae)


%% plot something
% figure()
% hold on
% scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),30,'bo','filled')
% scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),30,'ro','filled')
% scatter(X(50:138,1),X(50:138,2),'k*')      
% title('Sample points at iteration 89')