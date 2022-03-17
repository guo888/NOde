include("Forward.jl")

# Equation Parameters
# y' = f(y,x)
# -----------

# clear everything

nBP = 10000
# function f(y,x)
f = sin(x) ;
# Initial Condition
IC = -1.0;
# partial of function wrt y
pf =  0;
# exact solution!!
y =  -cos(x) ;  # 1./(1+exp(-x));

# Normal Distribution for weights
# rng("default")

# Number of Training Points
N = 100;

# Training Points
x= Array{Float64}(undef,N)
dx = 6*π/N
for j = 1:N
    x[j] = (j-1) * dx
end
x = 0:dx:6*π

# Network Parameters
# intial learning rate
eta = 0.01;
# drop rate
droprate = 1;
# hidden layer 神经数。
# size
H = 10; 
# biases
# b_H = normrnd(0,1,[H,1]);
b_H = randn(1,H)'
# weightss
# w_H = normrnd(0,1/sqrt(H),[H,1]);
w_H = randn(1,H)'
# output layer
# b_out = normrnd(0,1);
b_out = randn(1)[1]
# weights
# w_out = normrnd(0,1,[H,1]);
w_out = randn(1,H)'
# Variables for Plotting Output of Network
# output layer
a_out = zeros(N,1);

# feedforward over batches
for i = 1:N
    a_out[i] = forward(w_H,b_H,w_out,x(i));
end

# Plot Actual vs. ANN Initial Guess

plot(x,y)
plot(x,IC+x.*a_out)
xlabel("x")
ylabel("y")

title("Exact vs. ANN-initialized solution to y = y' ")
legend("Exact","ANN","location","northwest")

# backpropagation algorithm
for i = 1:nBP
    [w_H,b_H,w_out] = backPropagate(H,w_H,b_H, w_out,n,x,f,pf,IC,eta,droprate,i);
end
# feedforward over training inputs
for i = 1:n
    [a_H,z_H,a_out(i),z_out] = feedforward(w_H,b_H,w_out,x(i));
end

# Plot Actual vs. ANN Solution

plot(x,y)
plot(x,IC+x.*a_out)
xlabel("x")
ylabel("y")

title("Exact vs. ANN-computed solution to y' = y")
legend("Exact","ANN","location","northwest")

# Error Plot
n_err = N;
# sample
x_err = Array{Float64}(undef,N)
x_err = 0:1/n_err:1 
# x_err = linspace(0,1,n_err)";
a_out_err = zeros(n_err,1);
# feedforward over error-evaluating inputs
for i = 1:n_err
    [a_H,z_H,a_out_err(i),z_out] = feedforward(w_H,b_H,w_out,x_err(i));
end
# get errors
err = abs(y(x_err) - (IC + x_err.*a_out_err));

plot(x_err,err)
xlabel("x")
ylabel("error")

title("Absolute Error of ANN-computed solution to y"" = y")

# Extrapolation Plot
# m = 100;
ex = linspace(0,10,N)';
# feedforward over extrapolation points
for i = 1:m
    [a_H,z_H,a_out(i),z_out] = feedforward(w_H,b_H,w_out,ex(i));
end

plot(ex,y(ex))
plot(ex,IC+ex.*a_out)
xlabel("x")
ylabel("y")

title("Extrapolation of ANN-computed solution to y' = y")
legend("Exact","ANN","location","northwest")