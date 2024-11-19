function plotglobal(X,GlobalFig,iter)

figure(GlobalFig);
scatter(X(:,1),X(:,2),30,'bo','filled')
xlabel('x1');
ylabel('x2'); 
drawnow()

% plot gif
frame = getframe(GlobalFig);
im = frame2im(frame);
filename = "testAnimated.gif";
[A,map] = rgb2ind(im,256);
if iter == 1
    imwrite(A,map,filename,"gif","LoopCount",Inf,"DelayTime",0.2);
else
    imwrite(A,map,filename,"gif","WriteMode","append","DelayTime",0.2);
end
