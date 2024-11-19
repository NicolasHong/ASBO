function f = EIconstoadp_2(modelcon, x, fmin, modelobj)


[ypredCon, dyCon, mseCon]=predictor(x,modelcon);
sCon = sqrt(mseCon);

[ypredObj, dyObj, mseObj]=predictor(x,modelobj);
sObj = sqrt(mseObj);


% % A1: search outside the feasible region (from infeasible areas towards boundary)
% f1 = -(0-ypredCon)*normcdf((0-ypredCon)/sCon); % from outside
% f2 = 0;
% f3 = sCon*normpdf((0-ypredCon)/sCon);
% 
% % A2: search inside the feasible region (from feasible region towards boundary)
% % f1 = 0;
% % f2 = (0-U)*(1-normcdf((0-U)/s)); % from inside
% % f3 = s*normpdf((0-U)/s);
% 
% % A3: search near the boundary
% % f1 = 0;
% % f2 = 0;
% % f3 = 2*s*normpdf((0-U)/s);

% A4: search both from outside and inside
% f1 = -(0-ypredCon)*normcdf((0-ypredCon)/sCon); % from outside
% f2 = (0-ypredCon)*(1-normcdf((0-ypredCon)/sCon)); % from inside
% f3 = sCon*normpdf((0-ypredCon)/sCon);
% 
% % B1: more points near the boundary
% f = max([f1, f2, f3]);
% 
% % B2: less points near the boundary, more points in the feasible/infeasible
% % region. May cause under-estimation/over-estimation
% f = f1+f2+f3;


% EI = (fmin - ypred)*normcdf((fmin-ypred)/s) + s*normpdf((fmin-ypred)/s);
% EI = sCon*normpdf((0-ypredCon)/sCon);
% EI = f3;

% sCon1
% sCon2
% EI_con1 = sCon1*normpdf((0-ypredCon1)/sCon1)
% EI_con2 = sCon2*normpdf((0-ypredCon2)/sCon2)

EI = sCon*normpdf((0-ypredCon)/sCon);

if isempty(fmin)
    pModelObj = 1;
else
    pModelObj = normcdf((fmin - ypredObj)/sObj);
%     pModelObj = 1;
end

%f = -EI;
f = -EI*pModelObj;
end