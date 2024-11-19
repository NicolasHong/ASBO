function Xtest = generateXtest(d,ub,lb,n)

numTest = n;
Xlhs = lhsdesign(numTest, d); % test points
Xtest = zeros(numTest, d);
for i = 1:numTest
    Xtest(i,:) = lb + (ub - lb).*Xlhs(i,:);
end
