function [u]=testCase1_con(x,y)
if nargin == 1
  x1 = x(1);
  x2 = x(2);
else
  x1 = x;
  x2 = y;
end

c1=-2.*x1+x2-15;
c2=x1.^2/2+4.*x1-x2-5;
c3=-((x1-4).^2)/5-(x2.^2)/0.5+10;
u=max([c1;c2;c3]);
end