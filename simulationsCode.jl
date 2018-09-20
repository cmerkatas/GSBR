using Distributions
using Plots
using Polynomials
using Random
include("ToolBox.jl")
include("predictionGSBR.jl")
include("predictionDPR.jl")
### simulate from a cubic map with f₂₃ noise
theta =[0.05 2.55 0. -0.99]
n = 100; x0 = 1; seed = 4 # try seeds until you get bounded series
x = genData(n,theta,x0,seed)

plot(x=1:length(x),x,label="")

### set up sampler parameters for the GSBR model
# degree of modeling polynomial, max number of iterations, burnin period,
#         sampler seet, prediction horizon
deg,maxiter,burnin,seed,T = 5,250000,50000,1,5
@time thetas, zp, sx0,pred,clusters = predictionGSBR(x,deg,maxiter,burnin,seed,T,"/f23")

# estimates of coefficients
Θ̂₀ = mean(thetas[:,1])
Θ̂₁ = mean(thetas[:,2])
Θ̂₂ = mean(thetas[:,3])
Θ̂₃ = mean(thetas[:,4])
Θ̂₄ = mean(thetas[:,5])
Θ̂₅ = mean(thetas[:,6])


### set up parameters for the rDPR model
deg,maxiter,burnin,seed,T = 5,150000,50000,1,5
@time thetas, zp, sx0,pred,clusters = predictionDPR(x,deg,maxiter,burnin,seed,T,"/f23")

# estimates of coefficients
Θ̂₀ = mean(thetas[:,1])
Θ̂₁ = mean(thetas[:,2])
Θ̂₂ = mean(thetas[:,3])
Θ̂₃ = mean(thetas[:,4])
Θ̂₄ = mean(thetas[:,5])
Θ̂₅ = mean(thetas[:,6])
