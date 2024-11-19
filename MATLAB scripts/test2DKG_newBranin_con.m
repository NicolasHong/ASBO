function f = test2DKG_newBranin_con(x)
% lb = [-5, 0];
% ub = [10, 15];
% xGlob = [3.273, 0.0489];
% yGlob = -268.7879;

a=1;
b=5.1/(4*pi*pi);
c=5/pi;
d=6;
h=10;
ff=1/(8*pi);

x1 = x(1);
x2 = x(2);

c1 = a.*(x2-b.*x1.^2+c.*x1-d).^2+h.*(1-ff).*cos(x1)+h-5;

f = c1;

end