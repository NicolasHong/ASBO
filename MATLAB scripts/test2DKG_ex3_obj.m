function f=test2DKG_ex3_obj(x,y)

if nargin == 1
  x1 = x(1);
  x2 = x(2);
else
  x1 = x;
  x2 = y;
end

f=(x1-0.4)^2 + (x2-0.3)^2;