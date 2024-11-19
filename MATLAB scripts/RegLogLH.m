function f = RegLogLH(x,out,Bm,t)

A = x(1);
B = x(2);

% for i = 1:size(out)
%     P(i) = 1/(1+exp(A*out(i)+B*Bm(i)));
%     F(i) = t(i)*log(P(i))+(1-t(i))*log(1-P(i));
% end
% f = -sum(F);

fApB=out*A+B*Bm;   %a vector of the size of out
logF=log(1.0+exp(-abs(fApB)));
f = sum((t - (fApB < 0)) .* fApB + logF);




