function f = numStage2(stage_store)
n = length(stage_store);
f = 0;
for i = 1:n
    if stage_store(end+1-i)==2
        f = f+1;
    else
        break
    end
end
