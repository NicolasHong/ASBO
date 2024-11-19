% with 4 stages - local optimization stage added

clear; close all; clc;
fffthr = 0.05;   % threshold value to measure svm model changes
objthr = 0.0001;   % threshold value to measure fmin_diff

numRep = 100;
warning off

%% test problem
testObj = @testCase3_obj;
testCon = @testCase3_con;
numStr = 'Case3A';
lb = [-5, -20, -10, -12, -15];
ub = [12, 22, 10, 20, 18];
% xGlob = [3.273, 0.0489];
yGlob = 1.070;
numLoc = 200;
numAdppts = 100;
d = 5;

%% plot settings and plot original functions
xplot = gridsamp([lb; ub], 10);
K = size(xplot, 1);

stage_store_cell = cell(numRep,1);   
fmin_store_cell = cell(numRep,1);
for rep = 1:numRep
rng (rep)   % For reproducibility
rep
try
%% generate initial points

tinitial = tic;
tinitialcpu = cputime;

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

tinitialend = toc(tinitial)
tinitialcpuend = cputime - tinitialcpu

%% train model
ttrain = tic;
ttraincpu = cputime;
% Feasibility function
% RBF kernel bayesopt - check main_ConOpt_svm_20221027

% RBF kernel - grid sampling and no overfitting
cm = [0,1;1,0];
numGrid = 10;
hyperpts = gridsamplog([-3 -3; 3 3], numGrid); %range=[1e-3,1e3] for both scale and boxcon
hyperparaopt = selectRbfPara(numGrid,hyperpts,X,Ycon);
SVMModel = fitcsvm(X,Ycon,'Standardize',true,'KernelFunction','rbf',...
    'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));

% feasibility model outputs
SV_standardized = SVMModel.SupportVectors;
SV = [];
SV(:,1) = SV_standardized(:,1)*SVMModel.Sigma(1) + SVMModel.Mu(1);
SV(:,2) = SV_standardized(:,2)*SVMModel.Sigma(2) + SVMModel.Mu(2);
%SVL = SVMModel.SupportVectorLabels;

% objective function
[ModeltypeObj,thetaObj] = modelselectcon(Yobj,X);
modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);

ttrainend = toc(ttrain)
ttraincpuend = cputime-ttraincpu

%% plot initial models
for ii = 1:K 
    [yPredcon(ii, :),Score(ii,:)] = predict(SVMModel,xplot(ii, :));      
    [yPredobj(ii, :), ~,~] = predictor(xplot(ii, :), modelobj);
end   

% Calculate probability contour - Modified sigmoid
[A, B] = myfitPosterior(X,Ycon,XFeasSet,XInfeasSet,SVMModel); 

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

fffid=1;


%% Adaptive sampling
tadapt = tic;
tadaptcpu = cputime;

%while fmin_diff > objthr   % stopping criterion for entire algorithm
while iter<numAdppts
    
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


end


tadaptcpuend = cputime-tadaptcpu;

ttotalcpuend = tadaptcpuend+tinitialcpuend+ttraincpuend;


%% Results
% support vectors
SV_standardized = SVMModel.SupportVectors;
SV_final = [];
SV_final(:,1) = SV_standardized(:,1)*SVMModel.Sigma(1) + SVMModel.Mu(1);
SV_final(:,2) = SV_standardized(:,2)*SVMModel.Sigma(2) + SVMModel.Mu(2);
SVL = SVMModel.SupportVectorLabels;

% solution
Yobj_feas = Yobj(Ycon<=0);
[fmin,I] = min(Yobj_feas);
xmin = XFeasSet(I,:);


% plot fmin_store
figure('Visible', 'off')
plot([1:length(fmin_store)],fmin_store)
xlabel('Iteration','FontSize',15)
ylabel('Min obj','FontSize',15)
% title('fmin')

% plot stage
figure('Visible', 'off')
plot([1:length(stage_store)],stage_store)
xlabel('Iteration','FontSize',15)
ylabel('Stage index','FontSize',15)
% title({'Stage index';'1=Feasibility stage, 2=Optimization stage,3=Local refinement stage,4=Global exploration stage'})
yticks([1 2 3 4])
ylim([0.5 4.5])

stage_store_cell{rep,1} = stage_store;   
fmin_store_cell{rep,1} = fmin_store;   

catch exception
    stage_store_cell{rep,1} = exception;   
    fmin_store_cell{rep,1} = exception;   
end
filename1 = sprintf('%s_stage.mat', numStr);
filename2 = sprintf('%s_fmin.mat', numStr);
save(filename1, 'stage_store_cell');
save(filename2, 'fmin_store_cell');
end
