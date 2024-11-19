function f = EIobjstoadp_2(modelobj, x, fmin, modelcon, feasIndex)


% persistent n_calls; %<-this will be incrementing each time the function is called
% if isempty(n_calls)
%     n_calls=0;
% end
% global variable_tracking_matrix; %<-this is so you can see the global variable
% 
% 
% n_calls = n_calls + 1; %<-increments each time the objective_function completes
% variable_tracking_matrix(n_calls,1) = x(1); 
% variable_tracking_matrix(n_calls,2) = x(2);
% variable_tracking_matrix(n_calls,3) = x(3);
% variable_tracking_matrix(n_calls,4) = x(4);
% variable_tracking_matrix(n_calls,5) = x(5);
% variable_tracking_matrix(n_calls,6) = x(6);
% variable_tracking_matrix(n_calls,7) = x(7);
% variable_tracking_matrix(n_calls,8) = x(8);



[ypredObj, dyObj, mseObj]=predictor(x,modelobj);
sObj = sqrt(mseObj);
[ypredCon, dyCon, mseCon]=predictor(x,modelcon);
sCon = sqrt(mseCon);


% variable_tracking_matrix(n_calls,20) = mseCon2;
% variable_tracking_matrix(n_calls,9) = ypredObj;
% variable_tracking_matrix(n_calls,10) = sObj;
% variable_tracking_matrix(n_calls,11) = ypredCon1;
% variable_tracking_matrix(n_calls,12) = sCon1;
% variable_tracking_matrix(n_calls,13) = ypredCon2;
% variable_tracking_matrix(n_calls,14) = sCon2;

% correct EI
EI = (fmin - ypredObj)*normcdf((fmin-ypredObj)/sObj) + sObj*normpdf((fmin-ypredObj)/sObj);


%fake EI (LCB)
% LCB = ypredObj-4*sObj;
% if LCB > 98000
%     EI = 100000/(LCB-98000);    %because max EI = min LCB
% end


% fake EI (PI)
% EI = normcdf((fmin-ypredObj)/sObj);


% variable_tracking_matrix(n_calls,15) = EI;


if feasIndex > 0 % fmin is not obtained at a feasible location
    pModelCon = normcdf((feasIndex - ypredCon)/sCon);
else % fmin is obtained at a feasible location
%     variable_tracking_matrix(n_calls,16) = (0 - ypredCon1)/sCon1;
%     variable_tracking_matrix(n_calls,17) = (0 - ypredCon2)/sCon2;
     pModelCon = normcdf((0 - ypredCon)/sCon);
%     pModelCon = normcdf((fmin - yPredObj)/sObj);
%     pModelObj = 1;
end

% variable_tracking_matrix(n_calls,18) = pModelCon;

%f = -EI;
f = -EI*pModelCon;
% f = -sCon*pModelCon;
% f=LCB;

% variable_tracking_matrix(n_calls,19) = f;




end