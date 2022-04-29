function dsig(z)
    a = sig(z) .* (1.0 .-sig(z));
    return a 
end