function [u]=test2DKG_ex3_con(x,y)
%taken from Boukouvala and Ierapetritou CACE 36 (2012) 358-368
if nargin == 1
  x1 = x(1);
  x2 = x(2);
else
  x1 = x;
  x2 = y;
end

x1 = 15*x1 - 10;
x2 = 30*x2 - 15;

c1=-2.*x1+x2-15;
c2=x1.^2/2+4.*x1-x2-5;
c3=-((x1-4).^2)/5-(x2.^2)/0.5+10;
u=max([c1;c2;c3]);
end