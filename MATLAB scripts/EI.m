function y = EI(fmin,x,gprMdl)

[ypredObj,sObj,~] = predict(gprMdl,x);

y = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj) + sObj*normpdf((fmin-ypredObj)/sObj);
