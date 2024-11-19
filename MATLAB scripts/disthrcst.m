function thr = disthrcst(ub,lb)
%0.01*diagonal length of the problem
% lb = [-3, -2]; 
% ub = [3, 2];
diagl = norm(ub - lb);
thr = 1e-6*diagl;
