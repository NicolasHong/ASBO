% with 4 stages - local optimization stage added

clear; close all; clc;
fffthr = 0.05;   % threshold value to measure svm model changes
objthr = 0.0001;   % threshold value to measure fmin_diff

numRep = 100;
warning off

%% test problem
testObj = @testCase1_obj;
testCon = @testCase1_con;
numStr = 'Case1A';  % mewBraom
lb = [-10, -15];
ub = [5, 15];
xGlob = [-4, -6];
yGlob = 0;
numLoc = 50;  %init
numIter = 50;
d = 2;

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

figure('Visible', 'off')
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

stage_store_cell = cell(numRep,1);   
result_store_cell = cell(numRep,2);
fmin_store_cell = cell(numRep,1);

for rep=1:numRep
rng (rep)  % For reproducibility
rep    
%% generate initial points
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

figure('Visible', 'off')
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

figure('Visible', 'off')
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

ResultFig = figure('Visible', 'off','position', [200   200   1400   500]);
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
while fmin_diff > objthr && iter<=numIter-1   % stopping criterion for entire algorithm
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
            figure('Visible', 'off')
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
        end

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
        figure('Visible', 'off')
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
    % plotresult(ResultFig,stage_store,SVMModel,Yobj,Ycon,XFeasSet,xplot,...
    %     modelobj,x1Plot_surf, x2Plot_surf, yOrig_surf0con,...
    %     XiniFeasSet,XiniInfeasSet,XadpFeasSet,XadpInfeasSet,...
    %     xGlob,iter,numStr)

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
    
    figure('Visible', 'off','position', [693   217   861   708])
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
figure('Visible', 'off')
plot([1:length(fmin_store)],fmin_store)
xlabel('iteration')
ylabel('fmin')
title('fmin')

% plot stage
figure('Visible', 'off')
plot([1:length(stage_store)],stage_store)
xlabel('Iteration')
ylabel('Stage')
title({'Stage index';'1=Feasibility stage, 2=Optimization stage,3=Local refinement stage,4=Global exploration stage'})
ylim([0 4])

stage_store_cell{rep,1} = stage_store;   
fmin_store_cell{rep,1} = fmin_store;   
mae(rep) = fmin_diff;    


filename1 = sprintf('%s_stage.mat', numStr);
filename2 = sprintf('%s_fmin.mat', numStr);
save(filename1, 'stage_store_cell');
save(filename2, 'fmin_store_cell');


end

maeavg = mean(mae)
maesd = std(mae)



