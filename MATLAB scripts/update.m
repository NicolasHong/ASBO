function [X,Ycon,Yobj,XadpFeasSet,XFeasSet,XadpInfeasSet,XInfeasSet,...
    SVMModel,A_store,B_store,modelobj,fmin_store,xmin] = update...
    (Xf,testCon,testObj,X,XadpFeasSet,XFeasSet,XadpInfeasSet,XInfeasSet,...
    Ycon,Yobj,numGrid,hyperpts,cm,A_store,B_store,fmin_store)
                    
% update dataset
nn = size(Xf,1);
for i=1:nn
    Xff=Xf(i,:);
    X = [X;Xff];
    Yconf = testCon(Xff);
    Yobjf = testObj(Xff);
    if Yconf<=0
        Yconf = -1;
        XadpFeasSet(end+1,:) = Xff;
        XFeasSet(end+1,:) = Xff;
    else
        Yconf = 1;
        XadpInfeasSet(end+1,:) = Xff;
        XInfeasSet(end+1,:) = Xff;
    end
    Ycon = [Ycon;Yconf];
    Yobj = [Yobj;Yobjf];
end


% update feas model
    hyperparaopt = selectRbfPara(numGrid,hyperpts,X,Ycon);
    SVMModel = fitcsvm(X,Ycon,'Standardize',true,'KernelFunction','rbf',...
        'Cost',cm,'BoxConstraint',hyperparaopt(2),'KernelScale',hyperparaopt(1));     
    [A, B] = myfitPosterior(X,Ycon,XFeasSet,XInfeasSet,SVMModel);
    A_store(end+1) = A;
    B_store(end+1) = B;
    
% update obj model
    [ModeltypeObj,thetaObj] = modelselectcon(Yobj,X);
    modelobj = dacefit(X, Yobj, ModeltypeObj{1}, ModeltypeObj{2}, thetaObj, 0.1, 20);

% update fmin and xmin
    Yobj_feas = Yobj(Ycon<=0);
    X_feas = X(Ycon<=0,:);
    [fmin,I] = min(Yobj_feas);
    xmin = X_feas(I,:);
    fmin_store(end+1) = fmin;