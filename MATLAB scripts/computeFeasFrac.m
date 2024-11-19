function f = computeFeasFrac(SVMModel,xplot)

K = size(xplot, 1);
feasIndex = zeros(K,1);
for i=1:K
    x = xplot(i,:);
    [feasIndex(i),~] = predict(SVMModel,x);
end

numFeas = length(feasIndex(feasIndex==-1));
f = numFeas/K;
