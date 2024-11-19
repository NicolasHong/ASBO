function [A, B] = myfitPosterior(X,Ycon,XFeasSet,XInfeasSet,SVMModel)
for i = 1:size(X,1)
    [~,ScoreX(i,:)] = predict(SVMModel,X(i, :));
    dposX(i,1) = cald(X(i, :),XInfeasSet);
    dnegX(i,1) = cald(X(i, :),XFeasSet);
end
out = ScoreX(:,2);
target = Ycon;
prior0 = size(XFeasSet,1);     %negative value
prior1 = size(XInfeasSet,1);   %positive value


% use newton's method
%  [A,B] = platt_Basudhar(out,target,prior0,prior1,dposX,dnegX);


% optimize AB using fmincon with added constraints
% Bm = dnegX./(dposX+1e-10) - dposX./(dnegX+1e-10);
Bm = 1;
hiTarget=(prior1+1.0)/(prior1+2.0);
loTarget=1/(prior0+2.0);
t = (target>=0)*hiTarget + (target<0)*loTarget;
ABlb = [-100,-100];
% ABub = [0, 0];
ABub = [-3/min(max(out),-min(out)), 0];
Xtest = generateXtest(2,ABub,ABlb,10);
parfor i=1:size(Xtest,1)
    x = Xtest(i,:);
    fval(i) = RegLogLH(x,out,Bm,t);
end
[M,I] = min(fval);
ABini = Xtest(I,:);

myopt = optimoptions('fmincon', 'Display','off');  %or'Algorithm','sqp'
[ABopt,fval] = fmincon(@(x) RegLogLH(x,out,Bm,t),ABini,[],[],[],[],ABlb,ABub,...
                       [],myopt);
                   
                   
                   
% A = optimizableVariable('A',[-5,-3/min(max(out),-min(out))],'Transform','none');
% B = optimizableVariable('B',[-1,0],'Transform','none');
% minfn = @(x) RegLogLH(x,out,Bm,t);
% results = bayesopt(minfn,[A B],'IsObjectiveDeterministic',true,...
%     'AcquisitionFunctionName','expected-improvement-plus',...
%     'ExplorationRatio',0.5,...
%     'MaxObjectiveEvaluations',50,...
%     'UseParallel',false,...
%     'NumSeedPoints',10);
% z = bestPoint(results);
% ABopt = table2array(z);
                  
A = ABopt(1);
B = ABopt(2);