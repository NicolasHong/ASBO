function Y = test2DKG_gomez_con(x)
% lb = [-1, -1];
% ub = [1, 1];

x1 = x(1);
x2 = x(2);

% x1 = 0.1093;
% x2 = -0.6234;

% c1 = x2 + 0.38;
% c2 = -0.63 - x2;
% c3 = -0.02 - x1;
% c4 = x1 - 0.3;

c5 = -sin(4*pi*x1) + 2*(sin(2*pi*x2))^2;

% Y = max([c1, c2, c3, c4, c5]);
Y = c5;

end