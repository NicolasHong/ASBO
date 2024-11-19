% This is the main code to run WG flowsheet and compute feasibility
% function and obj function values

% The whole WG process is based on 2kg/hr flowrate, specific materials, 
% specific demand for energy/cost calculation. The flowrate is not a decision 
% variable as this model is not similar to the gPROMS model, which is based 
% on transport (ode,pde...), it is a population balance model based on 
% particle size distribution.

clear; clc; close all
% Just run some point
% Xini = [400, 0.66, 4, 70, 20];



%% Decision variable setting

% Decision variables: 
    % 1. Granulator speed - 400 in range [300,600] 
    % 2. Granulator L/S ratio - 0.66 in range [0.5,1]
    % 3. Dryer air velocity - 4 in range [2,8]
    % 4. Dryer air temperature - 70 in range [60,80]
    % 5. Dryer dry time - 20 in range [5,30]
    % 6. Mill rpm - 2400 in range [1500,3000]
X_label = ["Granulator speed (rpm)", "Granulator L/S ratio",...
           "Dryer air velocity (m/s)", "Dryer air temperature (°C)",...
           "Dryer dry time (min)", "Mill speed (rpm)"];
X_n = [400, 0.66, 4, 70, 20];   % nominal point(first 5 variables)
ub = [600, 1, 8, 80, 30];
lb = [300, 0.5, 2, 60, 5];
d = length(X_n);

k = 2000;
Xlhs = lhsdesign(k, d); % design points
Xini = zeros(k, d);
for i = 1:k
    Xini(i,:) = lb + (ub - lb).*Xlhs(i,:);
end
Ycon = zeros(k, 1);
Yobj = zeros(k, 1);



%% Run flowsheet

parfor ii=1:k
    
X = Xini(ii,:);
    
% parameters for WG model
%x_wg = [1e-7, 0.029, 0.7, 0.5, -4, -1, 20]; % does not work
% x_wg = [5.2179e-09,0.3928,0.0377,0.2968,0.3997,1.5720,23.3874];
x_wg = [1.15053624691180e-08,0.555892713646065,0.0889231010816998,0.451907862151614,0.173003375120492,1.45750554830032,31.2398888744810];
premixingTime = 120;
liquidAdditionTime = 300;
wetMassingTime = 0;
lsratio = X(2);
% Make sure sieveGrid has 180 and 1000 on the 3rd and 10th element,
% otherwise need to change yield calculation for granulator and mill.
% Also need to check PSD constraints since they depend on sieveGrid (c_180 etc.) 
%sieveGrid = [125, 250, 355, 500,600, 710, 850, 1000, 1100,1250, 1400, 1500, 1600, 1800, 2000,2200,2400,2600,2800];
% sieveGrid = [45, 90, 180, 250, 355, 500, 600, 710, 850, 1000, 1100,1250, 1400, 1500, 1600, 1800, 2000,2200,2400,2600,2800];
% sieveGrid =[90,180,250,500,850,1000,1400,2800,4750];
sieveGrid =[135   215 375 675 925 1200    2100    3775    4750];
%sieveGrid=[45.5 135 215 302.5 427.5 605 780 925 1200 1700 2400]; %this was from mill model diam_bin
ns = length(sieveGrid);
expdata = [];
rpm = X(1);
runflag = 'run';

% run the WG model
tic
[outpbm] = Model_function...
    (x_wg,premixingTime,liquidAdditionTime,wetMassingTime,...
    lsratio,sieveGrid,expdata,rpm,runflag);
toc

d10_wg = outpbm(1);
d90_wg = outpbm(2);
d50_wg = outpbm(3)*1000;
outDiams_135 = outpbm(4);
outDiams_215 = outpbm(5);
outDiams_375 = outpbm(6);
outDiams_675 = outpbm(7);
outDiams_925 = outpbm(8);
outDiams_1200 = outpbm(9);
outDiams_2100 = outpbm(10);
outDiams_3775 = outpbm(11);
outDiams_4750 = outpbm(12);
Use_E_wg = outpbm(end-2);
Tot_E_wg = outpbm(end-1);
MC_wg = outpbm(end);
yield_wg = outpbm(9)-outpbm(5);

% parameters for dryer model
conditions_dry = [X(4), MC_wg, X(3), 38, X(5)];

% run the dryer model
[MC_dry,Tot_E_dry] = generatedryerprediction_opt(conditions_dry);

% % parameters for mill model
% conditions_mill = [X(5) 450]; 
% p_batch = zeros(1,ns);
% p_batch(1) = outDiams_wg(1);
% for j=2:ns
%     p_batch(j)=outDiams_wg(j)-outDiams_wg(j-1);
% end
% p_batch(abs(p_batch)<1e-4)=0;     %assign very small values to be 0, otherwise t_inc in Mill model is too small and takes forever to run
% diam_bin=sieveGrid;
% 
% % run the mill model
% [distribution,tim,dia,fines,yield,oversize,Use_E_mill,Tot_E_mill,holdup,hp_dist,frac_cont,hp_t,time] = generateprediction_HillNg_cont_thy_ADUpdate(conditions_mill,p_batch,diam_bin);%, Dist,In_Mass)
% 

% Feasibility function 
% Constraints for feasibility can be yield, dryer MC, D50 (min or range?), 
% Below we calculate violation (positive violation means infeasible)

% c_yield = 0.60-yield;
c_MC_dry = MC_dry(end)-1;
% c_MC_wgl = 27.5-MC_wg;
% c_MC_wgu = MC_wg-50.5;
totalE = (Tot_E_wg * 1.67 * 308 * 10 * 4.8...
        + Tot_E_dry *7460/250 * 21074*1000/7460);
%        + Tot_E_mill * 308 *10 *2.5); 
c_totalE = totalE-5.3e8;
% c_d10l = 30.967-d10_wg;
% c_d10u = d10_wg-178.885;
% c_d50l = 204.064-d50_wg;
% c_d50u = d50_wg-534.135;
% c_d90l = 792.093-d90_wg;
% c_d90u = d90_wg-1279.556;
c_yield = 0.75-yield_wg;
% c_d50 = 270-d50_wg;
% C(i) = max([c_yield,c_MC_dry,c_d50]);
% c_180l = 0.1073132-outDiams_wg(2);
% c_180u = outDiams_wg(2)-0.49005375;
% c_250l = 0.19984102-outDiams_wg(3);
% c_250u = outDiams_wg(3)-0.57983598;
% c_500l = 0.46180758-outDiams_wg(4);
% c_500u = outDiams_wg(4)-0.81169723;
% c_850l = 0.7012987-outDiams_wg(5);
% c_850u = outDiams_wg(5)-0.98114368;
% c_1000l = 0.75242538-outDiams_wg(6);
% c_1000u = outDiams_wg(6)-1;
% c_1400l = 0.88001471-outDiams_wg(7);
% c_1400u = outDiams_wg(7)-1;


% Objective function
% total operation cost (material and utility)
materialCost = 648541/yield_wg; 
utilityCost = (totalE * 8 / 3600 * 0.065)/yield_wg;
totalCost = materialCost+utilityCost;  % cost/year
totalGWP = (55.27+ 806.42 + 0.016 + totalE * 8 / 3600*0.3755)/yield_wg; % kg CO2/year


[C,I] = max([c_yield,c_MC_dry,c_totalE]);


% [C(i),I(i)] = max([c_yield,c_MC_dry,c_MC_wgl,c_MC_wgu,c_totalE,c_180l,c_180u,...
%     c_250l,c_250u,c_500l,c_500u,c_850l,c_850u,c_1000l,c_1000u]);
% [C(i),I(i)] = max([c_MC_dry,c_d10l,c_d10u,...
%     c_d50l,c_d50u,c_d90l,c_d90u]);

Ycon(ii) = C;

Yobj(ii) =  totalCost+63.31/1000*totalGWP;

end

Yobj_feas = Yobj(Ycon<=0);
X_feas = Xini(Ycon<=0,:);
[fmin,I] = min(Yobj_feas);
Xmin = X_feas(I,:);