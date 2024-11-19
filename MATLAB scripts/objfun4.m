function f = objfun4(x,modelobj)

% Calculate EI
[ypredObj,~,~] = predictor(x, modelobj);
f = ypredObj;

end