function plotresult(ResultFig,stage_store,SVMModel,Yobj,Ycon,XFeasSet,xplot,...
                    modelobj,x1Plot_surf, x2Plot_surf, yOrig_surf0con,...
                    XiniFeasSet,XiniInfeasSet,XadpFeasSet,XadpInfeasSet,...
                    xGlob,iter,numStr)
                    

figure(ResultFig);

% stage plot
subplot(1,2,1)
plot([1:length(stage_store)],stage_store)
xlabel('Iteration')
ylabel('Stage')
title({'Stage index';'1=Feasibility stage, 2=Optimization stage,3=local refinement,4=Global exploration stage'})
ylim([0 4])
drawnow

% data&model plot
SV_standardized = SVMModel.SupportVectors;
SV_final = [];
SV_final(:,1) = SV_standardized(:,1)*SVMModel.Sigma(1) + SVMModel.Mu(1);
SV_final(:,2) = SV_standardized(:,2)*SVMModel.Sigma(2) + SVMModel.Mu(2);
%SVL = SVMModel.SupportVectorLabels;
Yobj_feas = Yobj(Ycon<=0);
[fmin,I] = min(Yobj_feas);
xmin = XFeasSet(I,:);
parfor i = 1:size(xplot,1)
[~,Score(i, :)] = predict(SVMModel,xplot(i, :));
[yPredobj(i, :), ~,~] = predictor(xplot(i, :), modelobj);
end
yPred_surf0con = reshape(Score(:,2), 100, 100);
yPred_surf0obj = reshape(yPredobj, 100, 100);

h2=subplot(1,2,2);
cla(h2)
hold on
scatter(XiniFeasSet(:,1),XiniFeasSet(:,2),30,'bo','filled')
scatter(XiniInfeasSet(:,1),XiniInfeasSet(:,2),30,'ro','filled')
scatter(xGlob(:,1),xGlob(:,2),100,'pentagram','filled')
[Ccon, Hcon] = contour(x1Plot_surf, x2Plot_surf, yPred_surf0con, [0, 0]);
set(Hcon, 'LineWidth', 1, 'LineColor', 'blue', 'LineStyle', '--');
contour(x1Plot_surf, x2Plot_surf, yPred_surf0obj);
scatter(SV_final(:,1),SV_final(:,2),70,'ks');
scatter(xmin(:,1),xmin(:,2),100,'ro');
if ~isnan(XadpFeasSet)
scatter(XadpFeasSet(:,1),XadpFeasSet(:,2),'b*') 
end
if ~isnan(XadpInfeasSet)
scatter(XadpInfeasSet(:,1),XadpInfeasSet(:,2),'r*')  
end
hold off
% legend('Feas fun','SVM for feas fun','GP for obj fun',...
%        'Ini feas','Ini infeas','Adp feas','Adp infeas',...
%        'Support vectors','True optimum','Optimal solution') 
title([numStr,' - Iteration = ',num2str(iter)])
drawnow()

% plot gif
frame = getframe(ResultFig);
im = frame2im(frame);
filename = "testAnimated.gif";
[A,map] = rgb2ind(im,256);
if iter == 1
    imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",0.2);
else
    imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",0.2);
end

