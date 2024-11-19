function setkSigmoidPara()

global kSigmoidGamma
global kSigmoidC
kSigmoidGamma = optimizableVariable('kSigmoidGamma',[1e-5,1e5],'Transform','log');
kSigmoidC = optimizableVariable('kSigmoidC',[1e-5,1e5],'Transform','log');

end

