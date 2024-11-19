function neardist = cald(x,X)
for i = 1:size(X,1)
    dist(i) = norm(x - X(i,:));

end

if size(X,1)>1
    dist(dist==0) = [];
end

neardist = min(dist);