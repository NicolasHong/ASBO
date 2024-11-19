function f = objfun2(x,modelobj,fmin)

% Calculate EI
[ypredObj,~,mseObj] = predictor(x, modelobj);
sObj = sqrt(mseObj);
   EI = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj) + sObj*normpdf((fmin-ypredObj)/sObj);
%   EI2 = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj);
%   LCB = ypredObj-0.0001*sObj;
   f=-EI;
%   f=-EI2;
%   f=LCB;