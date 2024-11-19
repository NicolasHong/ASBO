clear; close all; clc;

testObj = @compute_WG_obj;
testCon = @compute_WG_feas;
numStr = 'WG';
ub = [600, 1, 8, 80, 30];
lb = [300, 0.5, 2, 60, 5];
% yGlob = 7.7793e+05;
d = 5;
q = 5;
range = [lb;ub];
S = gridsamp(range, q);   % 5^5 = 3125 points
k = size(S,1);
X = S;

% start sampling
tinitialcpu = cputime;

Ycon = zeros(k, 1);
Yobj = zeros(k, 1);
for i = 1:k
    Ycon(i) = testCon(X(i, :));
    Yobj(i, :) = testObj(X(i, :));
end
Ycon(Ycon<=0)=-1;   %feasible
Ycon(Ycon>0)=1;    %infeasible

% categorize all sample points
XiniFeasSet = [];
XiniInfeasSet = [];
for i=1:k
    if Ycon(i)<=0
        XiniFeasSet(end+1, :) = X(i,:);
    else
        XiniInfeasSet(end+1, :) = X(i,:);
    end
end
XFeasSet = XiniFeasSet;
XInfeasSet = XiniInfeasSet;


tinitialcpuend = cputime - tinitialcpu