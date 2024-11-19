clear; close all; clc;

testObj = @test2DKG_newBranin_obj;
testCon = @test2DKG_newBranin_con;
numStr = 'newBranin';
lb = [-5, 0];
ub = [10, 15];
xGlob = [3.273, 0.0489];
yGlob = -268.7879;
numLoc = 49;
d = 2;

%% plot settings and plot original functions
xplot = gridsamp([lb; ub], 100);
K = size(xplot, 1);

parfor i = 1:K
    yOrigcon(i, :) = testCon(xplot(i, :)); 
    yOrigobj(i, :) = testObj(xplot(i, :));
end

xfeasSet = [];
for i = 1:K
    if yOrigcon(i)<=0
        xfeasSet(end+1, :) = xplot(i,:);
    end
end

x1Plot_surf = reshape(xplot(:, 1), 100, 100);
x2Plot_surf = reshape(xplot(:, 2), 100, 100);
yOrig_surf0con = reshape(yOrigcon, 100, 100);
yOrig_surf0obj = reshape(yOrigobj, 100, 100);

figure('position', [693   217   861   708])
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yOrig_surf0con, [0, 0]);
set(HH1, 'LineWidth', 1.5, 'LineColor', 'red', 'LineStyle', '-');
plot(xfeasSet(:,1),xfeasSet(:,2),'.','MarkerEdgeColor','r')
contour(x1Plot_surf, x2Plot_surf, yOrig_surf0obj, 'ShowText', 'on')
scatter(xGlob(:,1),xGlob(:,2),'fill')
hold off
title(numStr)
xlabel('x1');
ylabel('x2');
colorbar 

%% find radius for response surface
delta0 = 0.01;
x0 = [3.5-0.5956 1.5-0.5956];
A = [1 1; 1 -1; -1 1; -1 -1];

fun = @(delta)-delta;
nonlcon = @(delta)cons(delta,x0,testCon,A);

% max radius
r = fmincon(fun,delta0,[],[],[],[],[],[],nonlcon);

figure()
hold on
[CC1, HH1] = contour(x1Plot_surf, x2Plot_surf, yOrig_surf0con, [0, 0]);
set(HH1, 'LineWidth', 1.5, 'LineColor', 'blue', 'LineStyle', '-');
contour(x1Plot_surf, x2Plot_surf, yOrig_surf0obj, 'ShowText', 'on')
x1 = x0+A(1,:).*r;
x2 = x0+A(2,:).*r;
x3 = x0+A(3,:).*r;
x4 = x0+A(4,:).*r;
scatter(x0(1),x0(2),'ro')
scatter(x1(1),x1(2),'b*')
scatter(x2(1),x2(2),'b*')
scatter(x3(1),x3(2),'b*')
scatter(x4(1),x4(2),'b*')


%% construct response surface
dCC = ccdesign(d);
B = unique(dCC,'rows');
B = B./max(B,[],'all')

for i=1:size(B,1)
    x = x0+B(i,:)*r;
    xx(i,1) = x(1);
    yy(i,1) = x(2);
    zz(i,1)=testObj(x);
end

sf = fit([xx,yy],zz,'poly22');

figure()
plot(sf,[xx, yy],zz)


function [c,ceq] = cons(delta,x0,testCon,A)

x1 = x0+A(1,:).*delta;
x2 = x0+A(2,:).*delta;
x3 = x0+A(3,:).*delta;
x4 = x0+A(4,:).*delta;

c1 = testCon(x1);
c2 = testCon(x2);
c3 = testCon(x3);
c4 = testCon(x4);

c = max([c1 c2 c3 c4]);
ceq = [];
end