function f = objconfun1(x,SVMModelP,modelobj,fmin)

[ypredObj,~,mseObj] = predictor(x, modelobj);
sObj = sqrt(mseObj);
EI = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj) + sObj*normpdf((fmin-ypredObj)/sObj);

[~,PostProbs] = predict(SVMModelP,x); 
P1X = PostProbs(1);   % negative class

f = -EI*P1X;

