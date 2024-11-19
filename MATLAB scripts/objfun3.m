function f = objfun3(x,X,hyperparaopt)

N = size(X,1);
r = zeros(N,1);
scale = hyperparaopt(1);
for i=1:N
    dist = norm(x - X(i,:))/scale;
    r(i) = exp(-dist^2);           %rbf kernel
%      r(i) = x*X(i,:)';               %linear kernel

end

R = zeros(N);
for i = 1:N
    for j = 1:N
        dist = norm(X(i,:) - X(j,:))/scale;
        R(i,j) = exp(-dist^2);                %rbf kernel
%         R(i,j) = X(i,:)*X(j,:)';               %linear kernel
    end
end

% I = eye(N);
% miu = (10+N)*2^(-20);
% Rm = R+miu*I;
% C = chol(Rm);   % avoid singularity
% rt = C\r;
% uncty = 1-sum(rt.^2);
% uncty = 1 - r'*(Rm\r);  
% uncty = 1 - r'*(R\r);
uncty = 1-r'*pinv(R)*r;   % pseudo inverse
f = -uncty;          % minimize -uncertainty
 