function predictionGSBR(x,degree,maxiter,burnin,seed=1,T=0,filename = "/results")
  Random.seed!(seed)
  savelocation = string(pwd(), filename, "/gsbr_seed$seed")
  mkpath(savelocation)

  n = length(x)
  if T > 0
    append!(x, zeros(T))
    xpred = fill(Float64[],T)
    for t in 1:1:T
      xpred[t] = zeros(maxiter-burnin)
    end
    n_ = n + T
  else
    n_ = n
  end

  degree += 1
  theta = zeros(degree) # polynomial coeeficients vector
  sampledTheta = zeros(maxiter-burnin,degree) # matrix to store sample thetas
  thetaprev = copy(theta)
  epsj = 20.
  x0 = 0.0
  sampledX0 = zeros(maxiter-burnin)
  zp = zeros(maxiter-burnin)
  clusters = zeros(Int64,maxiter)
  L = zeros(maxiter)
  # MCMC OPTIONS
  ### specify geometric probability and the hyperparameters for the transformed gamma prior
  lambda = 0.5
  alambda,blambda= 1.,1.

  ### shape and rate for the gamma prior on precisions
  at, bt = 3,0.001

  # GSB meausre latent variables
  d = ones(Int64,n_)  # clustering variables
  N = copy(d)         # geometric slice variables

  ### begin MCMC
  for its in 1:1:maxiter

    M = maximum(N)
    w = zeros(M)
    tau = zeros(M)
    for j in 1:1:M
      w[j] = lambda * (1-lambda)^(j-1)
    end

    # sample precisions
    for j in 1:1:M
      term,counts = 0.,0.
      if d[1].==j
        counts += 1
        term += (x[1] - polyMap(theta,x0))^2
      end
      for i in 2:1:n_
        if d[i].==j
          counts += 1
          term += (x[i] - polyMap(theta,x[i-1]))^2
        end
      end
      tau[j] = rand(Gamma(at + 0.5counts,(bt + 0.5term)^(-1)))
    end

    # sample clustering variables
    nc0 = 0.
    for j in 1:1:N[1]
      nc0 += tau[j]^(0.5) * exp(-0.5*tau[j]*(x[1]-polyMap(theta,x0))^2)
    end
    rd0 = rand()
    prob0 = 0.
    for j in 1:1:N[1]
      prob0 += tau[j]^(0.5) * exp(-0.5*tau[j]*(x[1]-polyMap(theta,x0))^2) / nc0
      if rd0.<prob0
        d[1] = j
        break
      end
    end

    for i in 2:1:n_
      nc = 0.
      for j in 1:1:N[i]
        nc += tau[j]^(0.5) * exp(-0.5*tau[j]*(x[i]-polyMap(theta,x[i-1]))^2)
      end
      prob = 0.
      rd = rand()
      for j in 1:1:N[i]
        prob += tau[j]^(0.5) * exp(-0.5*tau[j]*(x[i]-polyMap(theta,x[i-1]))^2) / nc
        if rd.<prob
          d[i] = j
          break
        end
      end
    end

    clusters[its] = length(unique(d))
    ### update the geometric slice variables
    for i in 1:1:n_
      N[i] = tgeornd(lambda, d[i])
    end

    ### update probability lambda
    lambda = rand(Beta(alambda + 2n_, blambda + sum(N) - n_)) # Beta conjugate
    #lambda = synchronized(alambda, blambda, lambda, sum(N), n_)  # transformed gamma prior
    L[its] = lambda

    if its > burnin
      # sample from noise predictive
      W = cumsum(w)
      rp = rand()
      if rp.>W[end]
        zp[its-burnin] = rand(Normal(0, sqrt(1/rand(Gamma(at,1/bt)))))
      else
        for i in 1:1:length(W)
          if rp.<W[i]
            zp[its-burnin] = rand(Normal(0, sqrt(1/tau[i])))
            break
          end
        end
      end

      # sample the vector with the coefficients
      for j in 1:1:length(theta)
        thetaj = copy(theta)
        thetaj[j] = 0.

        tauj = tau[d[1]]*x0^(2(j-1))
        meanj = tau[d[1]]*x0^(j-1) * (x[1]-polyMap(thetaj,x0))
        for i in 2:1:n_
          tauj += tau[d[i]]*x[i-1]^(2(j-1))
          meanj += tau[d[i]]*x[i-1]^(j-1) * (x[i] - polyMap(thetaj,x[i-1]))
        end
        meanj /= tauj
        vj = -(2/tauj)*log(rand()) + (thetaprev[j]-meanj)^2
        theta[j] = rand(Uniform(max(-sqrt(vj)+meanj,-epsj), min(sqrt(vj)+meanj,epsj)))
      end
      thetaprev = copy(theta)
      sampledTheta[its-burnin,:] = copy(thetaprev)
      # sample x0
      aux = -(2.0/tau[d[1]])*log(rand()) + (x[1]-polyMap(theta,x0))^2
      poly_right = copy(theta)
      poly_right[1] = poly_right[1] - (x[1] - sqrt(aux))
      poly_right = poly_right./poly_right[end]
      roots_right = Polynomials.roots(Polynomial(poly_right))
      poly_left = copy(theta)
      poly_left[1] = poly_left[1] - (x[1] + sqrt(aux))
      poly_left = poly_left./poly_left[end]
      roots_left = Polynomials.roots(Polynomial(poly_left))
      allroots = [roots_left roots_right]
      idx = abs.(imag(allroots)).<1e-08
      realroots = sort(real(allroots[idx]))
      intervals = range_intersection(realroots,[-20. 20.])
      x0 = unifmixrnd(intervals)
      sampledX0[its-burnin] = x0

      # predict future unobserved observations directly
      # good results only for the noise process f₂
      if T > 0
        for j in 1:1:T-1
          ax1 = -(2/tau[d[n+j]])*log(rand()) + (x[n+j]-polyMap(theta,x[n+j-1]))^2
          ax2 = -(2/tau[d[n+j+1]])*log(rand()) + (x[n+j+1]-polyMap(theta,x[n+j]))^2

          xnm = polyMap(theta,x[n+j-1]) - sqrt(ax1)
          xnp = polyMap(theta,x[n+j-1]) + sqrt(ax1)

          poly_right_x = copy(theta)
          poly_right_x[1] = poly_right_x[1] - (x[n+j+1]-sqrt(ax2))
          poly_right_x = poly_right_x./poly_right_x[end]
          roots_right_x = Polynomials.roots(Polynomial(poly_right_x))

          poly_left_x = copy(theta)
          poly_left_x[1] = poly_left_x[1] - (x[n+j+1]+sqrt(ax2))
          poly_left_x = poly_left_x./poly_left_x[end]
          roots_left_x = Polynomials.roots(Polynomial(poly_left_x))

          allrootsx = [roots_left_x roots_right_x]
          idxx = abs.(imag(allrootsx)).<1e-06
          realrootsx = sort(real(allrootsx[idxx]))
          intervalsx = range_intersection(realrootsx,[xnm xnp])
          x[n+j] = unifmixrnd(intervalsx)
          xpred[j][its-burnin] = x[n+j]
        end
        new_mu = polyMap(theta,x[n+T-1])
        new_var = 1/tau[d[n+T]]
        x[n+T] = rand(Normal(new_mu, sqrt(new_var)))
        xpred[T][its-burnin] = x[n+T]
      end


    end


    if mod(its,10000)==0
      println("MCMC iterations: $its")
    end
  # end gibbs
  end

  ### write outputs to data files
  writedlm(string(savelocation, "/thetas.txt"), sampledTheta)
  writedlm(string(savelocation, "/X0.txt"), sampledX0)
  writedlm(string(savelocation, "/clusters.txt"), clusters)
  writedlm(string(savelocation, "/Prob.txt"), L)
  writedlm(string(savelocation, "/noise.txt"), zp)
  writedlm(string(savelocation, "/data.txt"), x)


  if T > 0
    for i in 1:1size(xpred)[1]
      writedlm(string(savelocation, "/xpred$i.txt"), xpred[i])
    end
    return sampledTheta, zp, sampledX0, xpred, clusters
  else
    return sampledTheta, zp, sampledX0, clusters
  end
end
