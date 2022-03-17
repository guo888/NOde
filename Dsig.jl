function dsig(z)
    a = sig(z) .* (1-sig(z));
    return a 
end