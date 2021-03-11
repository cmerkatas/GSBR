function predictionDPR(x,degree,maxiter,burnin,seed=1,T=0,filename = "/results")
  Random.seed!(seed)
  savelocation = string(pwd(), filename, "/dpr_seed$seed")
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
  C = zeros(maxiter)
  # MCMC OPTIONS

  ### initialize concentration parameter of the DP and specify prior hyperparameters
  c = 1.
  ac,bc= 1.,1.

  ### shape and rate for the gamma prior on precisions
  at, bt = 0.001,0.001

  ### DP meausre latent variables
  d = collect(1:1:n_) # clustering variables
  u = zeros(n_)       # slice variables

  ### begin MCMC
  for its in 1:1:maxiter

    M = maximum(d)
    v = zeros(M)
    w = zeros(M)
    tau = zeros(M)
    for j in 1:1:M
      flag1 = 0.0
      flag2 = 0.0
      for i in 1:1:n_
        if d[i].==j
          flag1 += 1.
        end
        if d[i].>j
          flag2 += 1.
        end
      end
      v[j] = rand(Beta(1 + flag1, c + flag2))
    end
    w[1] = v[1]
    for j in 2:1:M
      w[j] = w[j-1]*(1-v[j-1])*v[j]/(v[j-1])
    end

    # sample slice variables
    for i in 1:1:n_
      u[i] = rand(Uniform(0.0, w[d[i]]))
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

    # sample aditional number of weights
    sm, Mflag, iflag = 0, 0, 0
    sm = sum(w)
    ustar = minimum(u)
    while (sm < 1. - ustar)
      iflag = iflag + 1
      Mflag = M + iflag
      resize!(v, Mflag)
      resize!(w, Mflag)
      v[Mflag] = rand(Beta(1.0, c))
      w[Mflag] = w[Mflag - 1] * v[Mflag] * (1 - v[Mflag - 1]) / v[Mflag - 1]
      sm = sm + w[Mflag]
    end
    if Mflag > M
      append!(tau, rand(Gamma(at,1/bt), Mflag-M))
    end

    # sample clustering variables
    nc0 = 0.
    A1 = findall(u[1] .< w)
    for j in 1:1:length(A1)
      nc0 += tau[A1[j]]^(0.5) * exp(-0.5*tau[A1[j]]*(x[1]-polyMap(theta,x0))^2)
    end
    rd0 = rand()
    prob0 = 0.
    for j in 1:1:length(A1)
      prob0 += tau[A1[j]]^(0.5) * exp(-0.5*tau[A1[j]]*(x[1]-polyMap(theta,x0))^2) / nc0
      if rd0.<prob0
        d[1] = A1[j]
        break
      end
    end

    for i in 2:1:n_
      Ai = findall(u[i] .< w)
      nc = 0.
      for j in 1:1:length(Ai)
        nc += tau[Ai[j]]^(0.5) * exp(-0.5*tau[Ai[j]]*(x[i]-polyMap(theta,x[i-1]))^2)
      end
      prob = 0.
      rd = rand()
      for j in 1:1:length(Ai)
        prob += tau[Ai[j]]^(0.5) * exp(-0.5*tau[Ai[j]]*(x[i]-polyMap(theta,x[i-1]))^2) / nc
        if rd.<prob
          d[i] = Ai[j]
          break
        end
      end
    end

    clusters[its] = length(unique(d))

    # update concentration parameter
    c = learnc(ac,bc,c,n_,length(unique(d)))
    C[its] = c


    if its > burnin
      # sample from noise predictive
      W = cumsum(w)
      rp = rand()
      if rp.>W[end]
        zp[its-burnin] = rand(Normal(0., sqrt(1/rand(Gamma(at,1/bt)))))
      else
        for i in 1:1:length(W)
          if rp.<W[i]
            zp[its-burnin] = rand(Normal(0., sqrt(1/tau[i])))
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
      sampledTheta[its-burnin,:] = thetaprev
      # sample x0
      aux = -(2.0/tau[d[1]])*log(rand()) + (x[1]-polyMap(theta,x0))^2
      poly_right = copy(theta)
      poly_right[1] = poly_right[1] - (x[1] - sqrt(aux))
      poly_right = poly_right./poly_right[end]
      roots_right = Polynomials.roots(Poly(poly_right))
      poly_left = copy(theta)
      poly_left[1] = poly_left[1] - (x[1] + sqrt(aux))
      poly_left = poly_left./poly_left[end]
      roots_left = Polynomials.roots(Poly(poly_left))
      allroots = [roots_left roots_right]
      idx = abs.(imag(allroots)).<1e-06
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
          roots_right_x = Polynomials.roots(Poly(poly_right_x))

          poly_left_x = copy(theta)
          poly_left_x[1] = poly_left_x[1] - (x[n+j+1]+sqrt(ax2))
          poly_left_x = poly_left_x./poly_left_x[end]
          roots_left_x = Polynomials.roots(Poly(poly_left_x))

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
  writedlm(string(savelocation, "/C.txt"), C)
  writedlm(string(savelocation, "/noise.txt"), zp)


  if T > 0
    for i in 1:1:size(xpred)[1]
      writedlm(string(savelocation, "/xpred$i.txt"), xpred[i])
    end
    return sampledTheta, zp, sampledX0, xpred
  else
    return sampledTheta, zp, sampledX0
  end
end
