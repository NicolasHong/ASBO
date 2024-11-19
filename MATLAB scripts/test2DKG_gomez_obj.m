function Y = test2DKG_gomez_obj(x)
% lb = [-1, -1];
% ub = [1, 1];

x1 = x(1);
x2 = x(2);

% x1 = 0.1093;
% x2 = -0.6234;

Y = (4-2.1*x1^2+1/3*x1^4)*x1^2 + x1*x2 + (-4+4*x2^2)*x2^2;


end