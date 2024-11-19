%Select best modeltype for feasibility and objective
function [Modeltype2,theta2]=modelselectcon(CON,S)
theta=[0.1:0.1:10];

for i=1:length(theta)
   
theta0=theta(i);
lob=0.1;
upb=20;

[dmodelc0e, perff0e] = dacefit([S], CON, @regpoly0, @correxp, theta0);
mseC0e(i)=dmodelc0e.sigma2;


[dmodelc1e, perff1e] = dacefit([S], CON, @regpoly1, @correxp, theta0);
mseC1e(i)=dmodelc1e.sigma2;


 [dmodelc2e, perff2e] = dacefit([S], CON, @regpoly2, @correxp, theta0);
 mseC2e(i)=dmodelc2e.sigma2;

[dmodelc0l, perff0l] = dacefit([S], CON, @regpoly0, @corrlin, theta0);
mseC0l(i)=dmodelc0l.sigma2;

[dmodelc1l, perff1l] = dacefit([S], CON, @regpoly1, @corrlin, theta0);
mseC1l(i)=dmodelc1l.sigma2;

[dmodelc2l, perff2l] = dacefit([S], CON, @regpoly2, @corrlin, theta0);
 mseC2l(i)=dmodelc2l.sigma2;
 
 [dmodelc0g, perff0g] = dacefit([S], CON, @regpoly0, @corrgauss, theta0);
mseC0g(i)=dmodelc0g.sigma2;


[dmodelc1g, perff1g] = dacefit([S], CON, @regpoly1, @corrgauss, theta0);
mseC1g(i)=dmodelc1g.sigma2;

end

errC=[min(mseC0e);min(mseC1e);min(mseC2e);min(mseC0l);min(mseC1l);min(mseC2l);...
    min(mseC0g); min(mseC1g)];
errmin=min(errC);
if errmin==min(mseC0e)
    Modeltype2{1}=@regpoly0;
    Modeltype2{2}=@correxp;
    min2=mseC0e;
elseif errmin==min(mseC1e)
        Modeltype2{1}=@regpoly1;
        Modeltype2{2}=@correxp;
        min2=mseC1e;
elseif errmin==min(mseC2e)
                Modeltype2{1}=@regpoly2;
                Modeltype2{2}=@correxp;
                min2=mseC2e;
elseif errmin==min(mseC0l)
                            
                Modeltype2{1}=@regpoly0;
                Modeltype2{2}=@corrlin;
                min2=mseC0l;
elseif errmin==min(mseC1l)
                                
                Modeltype2{1}=@regpoly1;
                Modeltype2{2}=@corrlin;
                min2=mseC1l;
elseif errmin==min(mseC2l)
                                    
                Modeltype2{1}=@regpoly2;
                Modeltype2{2}=@corrlin;
                min2=mseC2l;

                
elseif errmin==min(mseC0g)
                                
                Modeltype2{1}=@regpoly0;
                Modeltype2{2}=@corrgauss;
                min2=mseC0g;
                
elseif errmin==min(mseC1g)

    Modeltype2{1}=@regpoly1;
    Modeltype2{2}=@corrgauss;
    min2=mseC1g;
    
    
%                     end
%                 end
%             end
%         end
%     end
end

for i=1:length(theta)
if min2(i)==min(min2)
   theta2=theta(i);
   break
end
end
    

end

