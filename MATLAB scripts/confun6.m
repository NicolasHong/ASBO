function f = confun6(x,PSVMModel)
% difference between class 1 and 2

[~,postprob] = predict(PSVMModel,x);
f = abs(postprob(1) - postprob(2));

end