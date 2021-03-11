using Distributions, StatsBase, StatsPlots
using Plots
using Polynomials
using Random
using DelimitedFiles
using KernelDensity
using LaTeXStrings
include("ToolBox.jl")
include("predictionGSBR.jl")
include("./data/datasets.jl")

# generate some data from chaotic logistic map
nf = 210
ntrain = 200
θ = [1.,0.,-1.71]
x₀ = .5
data = noisypolynomial(x₀, θ, noisemix2; n=nf, seed=11) # already "seeded"
plot(data)
### set up sampler parameters for the GSBR model
# degree of modeling polynomial, max number of iterations, burnin period,
#         sampler seet, prediction horizon
xtrain = data[2:ntrain]
ytest = data[ntrain+1:end]
deg,maxiter,burnin,seed,T = 5, 40000, 5000, 1, 10
@time thetas, zp, sx0, pred, clusters = predictionGSBR(x,deg,maxiter,burnin,seed,T,"/sims/logistic")

# estimates of coefficients

thetahat = mean(thetas, dims=1)
histogram(sx0)

zpl, zpr = -0.5, 0.5;
zp = zp[zp.>zpl];
zp = zp[zp.<zpr];

noiseplot = plot(kde(zp), color=:red, lw=1.5, label=L"\mathrm{est }\hat{f}(z)", grid=:false) #histogram(zp, bins=200, normalize=:pdf, color=:lavender, label="predictive");
x_range = range(zpl, stop=zpr, length=120);
ddnoise = noisemixdensity2.(x_range);
plot!(noiseplot, x_range, ddnoise, color=:black, lw=1.5, label=L"\mathrm{true } f(z)", ylim=(0,30), ylabel="pdf")

#std prediction plots
stdplt = scatter(data, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
plot!(stdplt, [ntrain], seriestype =:vline, colour = :green, linestyle =:dash, label = "Training Data End")

tsteps = 1:nf
fit = zeros(length(xtrain))
fit[1] = polyMap(thetahat, xtrain[1])
for i in 2:length(fit)
    fit[i] = polyMap(thetahat, fit[i-1])
end
plot!(stdplt, tsteps[2:ntrain], fit, alpha=0.4, colour =:blue, label = "fitted model")

allpredictions = hcat(pred...)
thinnedpredictions = allpredictions[1:10:end,:]

plot!(stdplt, tsteps[ntrain+1:end], mean(thinnedpredictions, dims=1)', color=:purple, ribbon=std(thinnedpredictions,dims=1)', alpha=0.4, label="prediction")

bestplt = scatter(tsteps[ntrain+1:end], ytest, colour = :blue, label = "Data", ylim=(-1.5, 2), grid=:false)
plot!(bestplt, tsteps[ntrain+1:end], mean(thinnedpredictions, dims=1)', color=:purple, ribbon=std(thinnedpredictions,dims=1)', alpha=0.4, label="prediction")

mean((mean(thinnedpredictions, dims=1)' .- ytest).^2)
