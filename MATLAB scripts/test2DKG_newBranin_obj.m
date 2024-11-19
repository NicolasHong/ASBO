function f = test2DKG_newBranin_obj(x)
% lb = [-5, 0];
% ub = [10, 15];
% xGlob = [3.273, 0.0489];
% yGlob = -268.7879;

x1 = x(1);
x2 = x(2);

f = -(x1-10)^2 - (x2-15)^2;

end