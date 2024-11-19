function S = gridsamplog(range, q)
lb = range(1,:);
ub = range(2,:);
d = length(lb);
if  length(q) == 1
    q = repmat(q,1,d); 
end


% Recursive computation
if  d > 1
  A = gridsamplog(range(:,2:end), q(2:end));  % Recursive call
  [m p] = size(A);   q = q(1);
  S = [zeros(m*q,1) repmat(A,q,1)];
  y = logspace(range(1,1),range(2,1), q);
  k = 1:m;
  for  i = 1 : q
    S(k,1) = repmat(y(i),m,1);  k = k + m;
  end
else    
  S = logspace(range(1,1),range(2,1), q).';
end
