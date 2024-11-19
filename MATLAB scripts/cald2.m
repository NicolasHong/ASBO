function neardist = cald2(x,X)
n=size(X,1);
dist = zeros(n,1);
for i = 1:n
    dist(i) = norm(x - X(i,:));

end

neardist = min(dist);