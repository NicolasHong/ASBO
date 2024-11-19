clc; 
clear; 
close all

tmark1 = tic; 
tstartcpu = cputime;
%% initialization     

% repeat
rep = 1; 
Yobj_rep_set = zeros(rep,1);
X_rep_set = zeros(rep,2);

% define lower bound and upper bound for input variables
X_label = ["Granulator speed (rpm)", "Granulator L/S ratio",...
           "Dryer air velocity (m/s)", "Dryer air temperature (°C)",...
           "Dryer dry time (min)", "Mill speed (rpm)"];
x0_mean = [400, 0.66, 4, 70, 20];   % nominal point(first 5 variables)
ub = [600, 1, 8, 80, 30];
lb = [300, 0.5, 2, 60, 5];
d = length(x0_mean);
numLoc = 50;

% define maximum iterations
iterMaxFeas = 250; % number of maximum iterations of adaptive sampling for feasibilty stage
iterMaxOpt = 250; % number of maximum iterations of adaptive sampling for optimization stgae
% fmin_sat = 95000;

% where to call simulation
testObj = @compute_WG_obj;
testCon = @compute_WG_feas;

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

% find feasible fmin among initial sample points (prepare to calculate EI)
feasSet = [];
xfeasSet = [];
for i = 1:k
    if Ycon(i)<=0
        feasSet(end+1, :) = Yobj(i);
        xfeasSet(end+1, :) = X(i,:);
    end
end
num_feas_ini_pts = length(feasSet);

if ~isempty(feasSet)
    [fmin, I] = min(feasSet);
    xfmin = xfeasSet(I, :);
else
    fmin = [];
    xfmin = [];
end

disp('Features of the dataset for building surrogate models:');
disp(['  There are ', num2str(num_feas_ini_pts), ' feasible points in the dataset.']);
disp(['  The one with lowest obj is x = ', num2str(xfmin)]);
disp(['  Its obj value is ', num2str(fmin)]);



%% build initial surrogate models

% select regression and correlation model

ModeltypeObj{1} = @regpoly0;
ModeltypeObj{2} = @correxp;
thetaObj = 0.5;   %??
ModeltypeCon{1} = @regpoly0;
ModeltypeCon{2} = @correxp;
thetaCon = 0.5;    %??


% fit kriging model
% tBuildSurrStart = tic;
modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);
modelcon = dacefit(X, Ycon, ModeltypeCon{1}, ModeltypeCon{2}, thetaCon, 0.1, 20);
% tBuildSurr = toc(tBuildSurrStart);



%% assign values to xmin_identified and fmin_identified
Yobj_feas = Yobj(Ycon<=0);
fmin = min(Yobj_feas);
xmin_identified = [];
fmin_identified = [];
feasIndex_identified = [];
if ~isempty(fmin) % there are feasible points in the initial data sample
    xmin_identified(end+1, :) = xfmin;
    fmin_identified(end+1, :) = fmin;
    feasIndex_identified(end+1, :) = -1; 
else % there are no feasible points in the initial data sample
    disp('no feasible pts in initial dataset');
    [Mtmp, Itmp] = min(Ycon);
    xmin_identified(end+1, :) = X(Itmp, :);
    fmin_identified(end+1, :) = Yobj(Itmp, :);
    feasIndex_identified(end+1, :) = Mtmp;
end


%% adaptive sampling of feaisiblity stage
disp('feasibility stage starts...');
iterFeas = 0;
EIconstoAdd = [];
N = 50*d;   % number of test points (restart of fmincon)(find initial point for fmincon)

while iterFeas < iterMaxFeas
    iterFeas = iterFeas+1;
    
    % generate test points by lhs
    testPtslhs = lhsdesign(N, d);  %generate N values for each of d variables
    for i = 1:N
        testPts(i,:) = lb + (ub - lb).*testPtslhs(i,:);
    end
    
    % find the point with min EIconstoadp among test points
    xminSet = zeros(N, d);
    fminSet = zeros(N, 1);
    for i = 1:N
        fminSet(i) = EIconstoadp_2(modelcon, testPts(i, :), fmin, modelobj);
    end
    [M, I] = min(fminSet);
    xAddini = testPts(I,:);
    
    % use xAddini as initial point for minimizing EIconstoadp by fmincon
     myopt = optimoptions('fmincon', 'Display','iter');
     
     [xAdd, EIconstoAdd(end+1)] = fmincon(@(x) EIconstoadp_2(modelcon, x, fmin, modelobj), xAddini,[], [], [], [], lb, ub, [], myopt);

    % display added point
    if ~isempty(fmin)
        disp(['KG_EIadp: iter ', num2str(iterFeas), ', xAdd = ', num2str(xAdd),...
        ', EIcon = ', num2str(EIconstoAdd(end)), ...
        ', fmin = ', num2str(fmin)]);
    else
        disp('no feasible pts');
        disp(['KG_EIadp: iter ', num2str(iterFeas), ', xAdd = ', num2str(xAdd),...
        ', EIcon = ', num2str(EIconstoAdd(end))]);
    end
    
    % update design points
    X = [X; xAdd];
    
    % update output
    YAddCon = testCon(X(end, :));
    YAddObj = testObj(X(end, :));
    Yobj = [Yobj; YAddObj];
    Ycon = [Ycon; YAddCon];

    k = length(X(:,1));
    
    
    % update fmin and measures
    if (~isempty(fmin) && YAddCon <= 0 && YAddObj < fmin) || (isempty(fmin) && YAddCon <= 0) % find a better feasible point
        fmin = YAddObj;
        xfmin = xAdd;
        % update measures
        xmin_identified(end+1, :) = xAdd;
        fmin_identified(end+1, :) = YAddObj;
        feasIndex_identified(end+1, :) = -1;
    elseif isempty(fmin) && YAddCon > 0 && YAddCon < feasIndex_identified(end,:) % find a better infesible point with less violations
        xmin_identified(end+1, :) = xAdd;
        fmin_identified(end+1, :) = YAddObj;
        feasIndex_identified(end+1, :) = YAddCon;
    else % no update
        xmin_identified(end+1, :) = xmin_identified(end, :);
        fmin_identified(end+1, :) = fmin_identified(end, :);
        feasIndex_identified(end+1, :) = feasIndex_identified(end, :);
    end
    
    
    
    % update model
modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);
modelcon = dacefit(X, Ycon, ModeltypeCon{1}, ModeltypeCon{2}, thetaCon, 0.1, 20);



end



%% adaptive sampling of optimization stage

iterOpt = 0;

EIobjstoAdd = [];

if isempty(fmin) % all sample points are infeasible
    disp('no feasible points in the sample. Sepcify a "fmin" with least feasibility violaitons')
    [Mcontmp, Icontmp] = min(Ycon);
    fmin = Yobj(Icontmp);
    xfmin = X(Icontmp,:);
    feasIndex = Mcontmp;
else
    feasIndex = -1;
end


% while fmin > fmin_sat
while iterOpt < iterMaxOpt 
    
    iterOpt = iterOpt+1;
    

    
    % generate test points by lhs
    testPtslhs = lhsdesign(N, d);
    for i = 1:N
        testPts(i,:) = lb + (ub - lb).*testPtslhs(i,:);
    end
    
    % find the new sample point
    xminSet = zeros(N, d);
    fminSet = zeros(N, 1);
    
    %myopt = optimoptions('fmincon', 'Display','off');
    myopt = optimoptions('fmincon', 'Display','iter','StepTolerance',1e-16);

    for i = 1:N
        fminSet(i) = EIobjstoadp_2(modelobj, testPts(i, :), fmin, modelcon, feasIndex);
    end
    [M, I] = min(fminSet);
    xAddini = testPts(I,:);
    [xAdd, EIobjstoAdd(end+1)] = fmincon(@(x) EIobjstoadp_2(modelobj, x, fmin, modelcon, feasIndex), xAddini,[], [], [], [], lb, ub, [], myopt);

    
     % display added point
    disp(['KG_EIadp: iter ', num2str(iterFeas+iterOpt), ', xAdd = ', num2str(xAdd),...
        ', EIobj = ', num2str(EIobjstoAdd(end)), ...
        ', fmin = ', num2str(fmin)]);
    
    % update design points
    X = [X; xAdd];
    
    % update output
    YAddCon = testCon(X(end, :));
    YAddObj = testObj(X(end, :));
    Yobj = [Yobj; YAddObj];
    Ycon = [Ycon; YAddCon];
    

    k = length(X(:,1));

    
    % update fmin and measures
    if (~isempty(fmin) && YAddCon <= 0 && YAddObj < fmin) || (isempty(fmin) && YAddCon <= 0) % find a better feasible point
        fmin = YAddObj;
        xfmin = xAdd;
        % update measures
        xmin_identified(end+1, :) = xAdd;
        fmin_identified(end+1, :) = YAddObj;
        feasIndex_identified(end+1, :) = -1;
    elseif isempty(fmin) && YAddCon > 0 && YAddCon < feasIndex_identified(end,:) % find a better infesible point with less violations
        xmin_identified(end+1, :) = xAdd;
        fmin_identified(end+1, :) = YAddObj;
        feasIndex_identified(end+1, :) = YAddCon;
    else % no update
        xmin_identified(end+1, :) = xmin_identified(end, :);
        fmin_identified(end+1, :) = fmin_identified(end, :);
        feasIndex_identified(end+1, :) = feasIndex_identified(end, :);
    end
    
    
    % update model    
    modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);
    modelcon = dacefit(X, Ycon, ModeltypeCon{1}, ModeltypeCon{2}, thetaCon, 0.1, 20);

    
end



%% return solution
disp([' Optimum solution X = ',num2str(xfmin),'; Yobj = ',num2str(fmin)]);

figure()
plot([1:length(fmin_identified)],fmin_identified)
xlabel('Iteration','FontSize',15)
ylabel('Min obj','FontSize',15)


%% total time consumed
tendcpu = cputime-tstartcpu;
% adaptive_sampling_time = toc(tmark2);
% disp('adaptive_sampling_time:');
% disp(datestr(datenum(0,0,0,0,0,adaptive_sampling_time),'HH:MM:SS'));
% total_time = toc(tmark1);
% disp('total time:');
% disp(datestr(datenum(0,0,0,0,0,total_time),'HH:MM:SS'));

%% plot EI
d1 = 1;  
d2 = 5;
Xsample_ub = ub([d1,d2]);
Xsample_lb = lb([d1,d2]);

samplesize = 100;
totalsize = samplesize^2;
Xsample = gridsamp([Xsample_lb;Xsample_ub], samplesize);

Xplot= zeros(totalsize,d);
for j=1:totalsize
    Xplot(j,:) = xAdd;
    Xplot(j,[d1 d2]) = Xsample(j,[1 2]);
end

for i = 1:totalsize
    EIobj(i) = EIobjstoadp_2(modelobj, Xplot(i, :), fmin, modelcon, feasIndex);
end

x1sPlot = reshape(Xsample(:,1),samplesize,samplesize);
x2sPlot = reshape(Xsample(:,2),samplesize,samplesize);
yPlot = reshape(EIobj,samplesize,samplesize);

figure()
contourf(x1sPlot, x2sPlot, yPlot,'Showtext', 'on')
xlabel(char(X_label(d1)),'FontSize',15)
ylabel(char(X_label(d2)),'FontSize',15)
c = colorbar;
c.Label.FontSize = 14;
