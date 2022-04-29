include("Forward.jl")
include("Backpropagate.jl")
using Plots
using Distributions
# Equation Parameters
# y' = f(y,x)
# -----------

# clear everything
function main()
    
    nBP = 1000
    # function f(y,x)
    f(y,x) = sin.(x) ;
    # Initial Condition
    IC = -1.0;
    # partial of function wrt y
    pf(x) =  0;
    # exact solution!!
    y(x) =  -cos.(x) ;  # 1./(1+exp(-x));

    # Normal Distribution for weights
    # rng("default")

    # Number of Training Points
    N = 100;

    # Training Points
    x= Array{Float64}(undef,N)
    dx = 0.1 #6*π/N
    for j = 1:N
        x[j] = (j-1) * dx
    end
    # x = 0:dx:6*π

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
    # b_H .= 1.0
    # weightss
    # w_H = normrnd(0,1/sqrt(H),[H,1]);
    w_H = rand(Normal(0,1/sqrt(H)),H) #(1,H)'
    # w_H .= 1.0
    # output layer
    # b_out = normrnd(0,1);
    b_out = randn(1)[1]
    # b_out = 1.0
    # weights
    # w_out = normrnd(0,1,[H,1]);
    w_out = randn(1,H)'
    # w_out .= 1.0 
    # Variables for Plotting Output of Network
    # output layer
    a_out = zeros(N,1);

    # feedforward over batches
    for i = 1:N
        temp1,temp2,a_out[i],temp3 = feedForward(w_H,b_H,w_out,x[i]);
    end
    println("a_out= ",a_out)
    # Plot Actual vs. ANN Initial Guess
    # p0 = plot(1)
    # y(x) =  -cos.(x) ; 
    # p1=plot!(x,y)
    # plot!(p1,x,IC .+ x .* a_out)
    # xlabel!("x")
    # ylabel!("y")

    # title!("Exact vs. ANN-initialized solution to y = y' ")
    # savefig("p1.png")
    # display(p1)
    # # legend!("Exact","ANN","location","northwest")
    # exit()
    ############################## mingtian jixu
    # backpropagation algorithm
    for i = 1:nBP
        w_H_N,b_H_N,w_out_N = backPropagate(H,w_H,b_H, w_out,N,x,f,pf,IC,eta,droprate,i);
        w_H,b_H,w_out = w_H_N,b_H_N,w_out_N 
    end
    # feedforward over training inputs
    for i = 1:N
        a_H,z_H,a_out[i],z_out = feedForward(w_H,b_H,w_out,x[i]);
    end

    # Plot Actual vs. ANN Solution       
    p0 = plot(1)
    p2 = plot!(p0,x,y(x),color=:red)
    p2 = plot!(p2,x,IC .+ x.*a_out)
    # savefig("curve.png")
    xlabel!("x")
    ylabel!("y")
    display(p2)
    savefig("p2.png")
    title!("Exact vs. ANN-computed solution to y' = y")
    # exit()
    # legend!("Exact","ANN","location","northwest")
# ######################################
#     # Error Plot
    n_err = N;
    # sample
    x_err = Array{Float64}(undef,N)
    x_err .= x
    # x_err = 0:1/n_err:1 -1/n_err
    # x_err = linspace(0,1,n_err)";
    a_out_err = zeros(n_err,1);
    # feedforward over error-evaluating inputs
    for i = 1:n_err
        a_H,z_H,a_out_err[i],z_out = feedForward(w_H,b_H,w_out,x_err[i]);
    end
    # get errors
    err = abs.(y(x_err) .- (IC .+ x_err.*a_out_err));

    plot!(x_err,err)
    xlabel!("x")
    ylabel!("error")

    # title!("Absolute Error of ANN-computed solution to y"" = y")
##################################################################
# #     # Extrapolation Plot
#     m = 2* N;
#     # ex = Array{Float64}(undef,N)
#     # ex .= x
#     # ex = linspace(0,10,N)';
#     ex = range(1, 10, length=m)
#     a_out = zeros(m,1);
#     # feedforward over extrapolation points
#     for i = 1:m
#         a_H,z_H,a_out[i],z_out = feedForward(w_H,b_H,w_out,ex[i]);
#     end

#     # plot!(ex,y(ex))
#     p2 = plot!(p2,ex,IC .+ ex.*a_out)
#     xlabel!("x")
#     ylabel!("y")

#     title!("Extrapolation of ANN-computed solution to y' = y")
#     # legend("Exact","ANN","location","northwest")
# ######################################
    # return w_H
end


##############################


main()