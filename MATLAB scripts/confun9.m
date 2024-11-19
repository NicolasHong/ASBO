function f = confun9(x,X,A,B,SVMModel,d_scale,score_scale)
%Reference - Song, C., Shafieezadeh, A., & Xiao, R. (2022). High-Dimensional Reliability Analysis with Error-Guided Active-Learning Probabilistic Support Vector Machine: Application to Wind-Reliability  Analysis of Transmission Towers. Journal of Structural Engineering, 148(5). doi:10.1061/(asce)st.1943-541x.0003332

N = size(X,1);
dist = zeros(N,1);
for i = 1:N
    dist(i) = norm(x - X(i,:));
end
dist_min = min(dist)/d_scale;

[~,scorex] = predict(SVMModel,x);
% scorexn = scorex(1)/score_scale;

% PostProbPos = 1./(1+exp(A.*scorex(2)+B));

if scorex(2)>=0
    PostProbPos = 1./(1+1+scorex(2));
elseif scorex(2)<0
    PostProbPos = 1./(1+exp(scorex(2)));
end

PostProbNeg = 1-PostProbPos;


probs = [PostProbPos PostProbNeg];
Pwse = min(probs);

F = Pwse*dist_min;
f = -F;


% f=scorexn/dist_min;
end