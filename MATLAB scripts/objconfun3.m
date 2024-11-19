function f = objconfun3(x,A,B,XFeasSet,XInfeasSet,SVMModel,modelobj,fmin)

% Calculate EI
[ypredObj,~,mseObj] = predictor(x, modelobj);
sObj = sqrt(mseObj);
EI = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj) + sObj*normpdf((fmin-ypredObj)/sObj);

%Calculate P(-1|x)
dpos = cald(x,XInfeasSet);
dneg = cald(x,XFeasSet);
Bm = dneg/(dpos+1e-10) - dpos/(dneg+1e-10);
[~,Score] = predict(SVMModel,x);
PostProbsCal = 1./(1+exp(A.*Score(2)+B*Bm));
P1X = 1-PostProbsCal; % negative class
  
f = -(fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj)*P1X...
    -sObj*normpdf((fmin-ypredObj)/sObj);   % min f is the next selected point