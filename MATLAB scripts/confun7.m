function [c,ceq] = confun7(x,A,B,XFeasSet,XInfeasSet,SVMModel)

dpos = cald(x,XInfeasSet);
dneg = cald(x,XFeasSet);
Bm = dneg/(dpos+1e-10) - dpos/(dneg+1e-10);
[~,Score] = predict(SVMModel,x);

%Calculate P(-1|x)
% PostProbsCal = 1./(1+exp(A.*Score(2)+B*Bm));
% P1X = 1-PostProbsCal; % negative class
% c = 0.5-P1X;

%calculate classification score
%Score(1) should > 0
c = -Score(1);


ceq=[];




