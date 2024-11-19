function [f] = calcCorr(lb, ub, testCon, SVMModel,nTest)

% nTest = 100;
testPts = gridsamp([lb; ub],nTest); %test points

nCorFeas = 0;
nIncorFeas = 0;

nCorInfeas = 0;
nIncorInfeas = 0;

for i = 1:length(testPts(:,1))
    
    original = testCon(testPts(i,:));
    pred = predict(SVMModel,testPts(i,:));
    
    if original <= 0 && pred <= 0
        nCorFeas = nCorFeas + 1;
    elseif original > 0 && pred <= 0
        nIncorFeas = nIncorFeas + 1;
    elseif original > 0 && pred > 0
        nCorInfeas = nCorInfeas + 1;
    else
        nIncorInfeas = nIncorInfeas + 1;
    end

end

corrFeas_perc = nCorFeas/(nCorFeas + nIncorInfeas);
corrInfeas_perc = nCorInfeas/(nCorInfeas + nIncorFeas);
overEsti_perc = nIncorFeas/(nCorFeas + nIncorFeas);

f(1) = corrFeas_perc;
f(2) = corrInfeas_perc;
f(3) = overEsti_perc;