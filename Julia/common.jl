function loadData(model::Dict{Symbol, Any})

    if model[:dataset] == "ConcreteStrength"
          C = Matrix(CSV.read("Concrete_Data_Yeh.csv", DataFrame))
          x = C[:,1:end-1]
          y = C[:,end]

    end

    if model[:kernelVersion] == "individual"
        row, col = size(x)
        model[:initialParams] = zeros(col+1)
    elseif model[:kernelVersion] == "individualScale"
        row, col = size(x)
        model[:initialParams] = zeros((col*2) + 1)
    end

    model[:x]   = x
    model[:y]   = y
    row, col    = size(model[:x])
    c           = round(Int, row*model[:testSize])
    perm        = Random.shuffle(1:row)
    idxTest     = perm[1:c]
    idxTraining = perm[c+1:end]

    model[:X]   = model[:x][idxTraining,:]
    model[:Y]   = model[:y][idxTraining,:]
    model[:Xtest] = model[:x][idxTest,:]
    model[:Ytest] = model[:y][idxTest,:]
    
    model[:Y], model[:muY], model[:stdY] = zscore(model[:Y])
    model[:X], model[:muX], model[:stdX] = zscore(model[:X])
    model[:Ytest] = normalize(model[:Ytest], model[:muY], model[:stdY])
    model[:Xtest] = normalize(model[:Xtest], model[:muX], model[:stdX])

    if model[:kernelVersion] == "withPCA"
        T, P = compute_pca(model[:X])
        Ttest  = model[:Xtest] * P

        model[:Xtest] = Ttest[:,1:model[:noPCs]]
        model[:X]     = T[:,1:model[:noPCs]]
        model[:kernelVersion] = "individual"
    end

       
    if isa(model[:X], Tuple)
        model[:X] = hcat(model[:X]...)  
    end

    model[:X] = Array{Float64}(model[:X])

    if size(model[:X], 1) == 0 || size(model[:X], 2) == 0
        error("model[:X] has zero rows or columns, which could cause indexing errors.")
    end

    return model
end

# Function to apply Savitzky-Golay filter to a row
function apply_savgol_filter(row, window_size, polynomial_order, deriv)
    sg = savitzky_golay(row, window_size, polynomial_order, deriv=deriv)
    return sg.y  # Extract the filtered signal
end

function zscore(data)
    mu      = mean(data, dims=1)
    sigma   = std(data, dims=1)
    normalized_data = (data .- mu) ./ sigma
    return normalized_data, mu, sigma
end

function normalize(data, mu, sigma)
    return (data .- mu) ./ sigma
end

     
# Function to apply z-score normalization to each row
function zscore_row(row)
    μ = mean(row)
    σ = std(row)
    return (row .- μ) ./ σ
end

function normDiff(x, y)
    sqx = x .^ 2
    sqy = y .^ 2
    nsx = sum(sqx, dims=2)
    nsy = sum(sqy, dims=2)
    nsx = vec(nsx)
    nsy = vec(nsy)
    innerMat = x * y'
    nD = -2 .* innerMat .+ nsx .+ nsy'
    return nD
end

function moving_average(x, window_size)
    return [mean(x[max(1, i-window_size+1):i]) for i in 1:length(x)]
end

function kernelRBF2(X::Matrix{Float64}, params, model::Dict{Symbol,Any}, state)
    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row, col))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-08.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        K = zeros(noRows, noRows)
        
        for i = 1:noVars
            ND = normDiff(X[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row, col))
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-8.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (params[i + noVars] .* 1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-08.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(X, X)
        row, col = size(ND)
        d   = sqrt.(ND + 1e-08.*ones(row,col))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ params[10] * (1. ./ (ones(row, col) + (ND ./ params[9]^2)))

    else
        ND = normDiff(X, X)
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-08.*ones(row,col))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end
    
    if state == "training"
        nl   = size(K, 1)
        oneN = ones(nl, nl) / nl
        K    = K .- oneN * K .- K * oneN .+ oneN * K * oneN .+ params[end] * I(nl)
    else 
        nl   = size(K, 1)
        K   =  K .+ params[end] * I(nl)
    end
    return K
end

function kernelRBFTest(X::Matrix{Float64}, XTest::Matrix{Float64}, params, model::Dict{Symbol,Any})

    KCal = kernelRBF2(X, params, model, "test")

    if model[:kernelVersion] == "individual"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row, col))
                K = K + exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-09.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end

    elseif model[:kernelVersion] == "individualScale"
        noRows, noVars = size(X)
        noRowsT, noVarsT = size(XTest)

        K = zeros(noRowsT, noRows)
        
        for i = 1:noVars
            ND = normDiff(XTest[:,i], X[:,i])
            if model[:kernelType] == "gaussian"
                K = K + params[i + noVars] .* exp.(-ND)/(2*params[i]^2)
            elseif model[:kernelType] == "matern1/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row, col))
                K = K + params[i + noVars] .* exp.(-d/params[i])
            elseif model[:kernelType] == "matern3/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-9.*ones(row,col))
                sqrt3_d = sqrt(3) * d / params[i]
                K = K + params[i + noVars] .* (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
            elseif model[:kernelType] == "matern5/2"
                row, col = size(ND)
                d = sqrt.(ND + 1e-09.*ones(row,col))
                sqrt5_d = sqrt(5)*ones(row,col) + d / params[i]
                term2 = (5.0 * ND) / (3.0 * params[i]^2)
                K = K + params[i + noVars] .* (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
            elseif model[:kernelType] == "cauchy"
                row, col = size(ND)
                K = K + params[i + noVars] .* 1. ./ (ones(row, col) + (ND ./ params[i]^2))
            else 
                println("The kernel method is not available for individual calculations.")
            end
        end
    
    elseif model[:kernelVersion] == "family"
        ND  = normDiff(XTest, X)
        row, col = size(ND)
        d   = sqrt.(ND + 1e-08.*ones(row,col))
        K   = params[2] * exp.(-ND/(2*params[1]^2))
        K   = K .+ params[3] * exp.(-d/params[4])
        sqrt3_d = sqrt(3) * d / params[6]
        K   = K .+ params[5] * ((1. .+ sqrt3_d) .* exp.(-sqrt3_d))
        sqrt5_d = sqrt(5)*ones(row,col) + d / params[8]
        term2 = (5.0 * ND) / (3.0 * params[8]^2)
        K   = K .+ params[7] * ((1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d))
        K   = K .+ 1. ./ (ones(row, col) + (ND ./ params[9]^2))

    else
        ND = normDiff(XTest, X)
        if model[:kernelType] == "gaussian"
            K = exp.(-ND/(2*params[1]^2))
        elseif model[:kernelType] == "matern1/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            K = exp.(-d/params[1])
        elseif model[:kernelType] == "matern3/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-9.*ones(row,col))
            sqrt3_d = sqrt(3) * d / params[1]
            K = (1. .+ sqrt3_d) .* exp.(-sqrt3_d)
        elseif model[:kernelType] == "matern5/2"
            row, col = size(ND)
            d = sqrt.(ND + 1e-09.*ones(row,col))
            sqrt5_d = sqrt(5)*ones(row,col) + d / params[1]
            term2 = (5.0 * ND) / (3.0 * params[1]^2)
            K = (1. .+ sqrt5_d .+ term2) .* exp.(-sqrt5_d)
        elseif model[:kernelType] == "cauchy"
            row, col = size(ND)
            K = 1. ./ (ones(row, col) + (ND ./ params[1]^2))
        else
            println("Error! Model kerneltype incorrecty")
        end

    end

    n       = size(KCal, 1)
    oneN    = ones(n, n)/n
    nTest   = size(XTest, 1)
    oneNTest = ones(nTest, n)/n
    KTest       = K .- oneNTest * KCal .- K * oneN .+ oneNTest * KCal * oneN 
    return KTest
end

function compute_pca(x::Matrix{Float64})
    m, n = size(x)
    # Check for NaN or infinite values
    if any(isnan, x) || any(isinf, x)
        error("Matrix contains NaN or infinite values")
    end

    if m < n
        p_adj, s, u = svd(x')
    else
        u, s, p_adj = svd(x)
    end
    t = u * Diagonal(s)
    return t, Matrix(p_adj)
end

function compute_pls(X::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    m, nx = size(X)
    m, ny = size(Y)
    T = zeros(m, 0) 
    P = zeros(nx, 0)
    Q = zeros(ny, 0)
    W = zeros(nx, 0)
    U = zeros(m, 0)

    for i in 1:dim
        C = Y' * X
        A, D = compute_pca(C)
        w = D[:,1]

        t = X * w
        q = (Y' * t) / (t' * t)
        u = (Y * q) / (q' * q)

        T = hcat(T, t)
        Q = hcat(Q, q)
        W = hcat(W, w)
        U = hcat(U, u)

        p = (X' * t) / (t' * t)
        P = hcat(P, p)

        X = X - t * p'
        Y = Y - t * q'
    end
    return T, P, Q, W, U
end

function compute_pls2(X::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    m, nx = size(X)
    m, ny = size(Y)
    T = zeros(m, 0) 
    U = zeros(m, 0)
    K = X

    if model[:PLS] == "SIMPLS"

    for i in 1:dim
        t0 = K * Y
        
        t = t0./sqrt(sum(t0 .* t0))
        u = Y * (Y' * t)

        T = hcat(T, t)
        U = hcat(U, u)

        K       = K  - t * (t' * K)
        Y       = Y  - t * (t' * Y)
    end

    elseif model[:PLS] == "NIPALS"
    tol = sqrt(eps())
    maxit = 100

    n, zp = size(X)
    q = size(Y, 2)

    for a = 1:dim
        if q == 1
            t0 = K * Y
            t = t0./sqrt(sum(t0 .* t0))
            c = Y' * t
            u = Y * c
            u2 = u./sqrt(sum(u .* u))
        else
            u = Y[:, 1]
            ztol = 1.0
            iter = 1

            while ztol > tol && iter <= maxit
                t = K * u
                t ./= sqrt(sum(t .* t))
                c = Y' * t
                zu = Y * c
                zu ./= sqrt(sum(zu .* zu))
                ztol = norm(u - zu)
                u = zu
                iter += 1
            end
        end

        z = I - t * t'
        K = z * K * z'
        Y -= t * c'

        T = hcat(T,t)
        #C[:, a] = c
        U = hcat(U,u2)
    end

    
    end
    return T, U
end

function compute_kpls(K::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    if model[:PLS] == "NIPALS" || model[:PLS] == "SIMPLS"
        T, U    = compute_pls2(K, Y, dim)
        B       = U * ((T' * K * U))^(-1)*T'*Y;
    elseif model[:PLS] == "RBF"
        T, P, Q, W, U = compute_pls(K, Y, dim)
        B = W * inv(P' * W)* Q'
    elseif model[:PLS] == "PCR"
        T, P = compute_pca(K)               # 1. Compute KPCA
        Bsmall = T[:,1:model[:dim]]\Y       # 2. Weights in KPC space
        B = P[:,1:model[:dim]] * Bsmall     # 3. Weights in K space
    end    

    return B # W is only utilized in conformal inference
end 

function compute_kpls2(K::Matrix{Float64}, Y::Matrix{Float64}, dim::Int)
    T, U = compute_pls2(K, Y, dim)
    row, col = size(T)

    R = zeros(col,1)
    w = zeros(col,1)
    TSS = sum((K .- mean(K, dims=1)).^2)
    for i = 1:col
        ti = T[:,i]
        #println(ti)
        ip = transpose(ti)*ti
        iv = inv(ip)
        ipp = iv*transpose(ti)
        s5 = ipp * K
        ti = reshape(ti, row, 1)
        s5 = reshape(s5, row, 1)
        Kr =  ti * s5'
        RSS = sum((Kr - K).^2)
        R[i] = 1 - RSS/TSS
        println(R[i])
    end

    for i = 1:col
        w[i] = R[i]/sum(R)
    end

    B = U * inv(T' * K * U) * T' * Y
    return B, T, U, w # W is only utilized in conformal inference
end 

function sample(n::Int, sp::Float64)
    ns    = ceil(Int, n*sp)
    perm  = Random.shuffle(1:n)
    idxS = perm[1:ns]
    return idxS
end

function rho_average(params::Vector{Float64}, X::Matrix{Float64}, Y::Matrix{Float64}, model::Dict{Symbol,Any})
    #println(size(Y))
    #println(size(X))

    XBatch  = X[model[:idxBatch], :]
    YBatch  = Y[model[:idxBatch], :]

    loss    = zeros(1,1)

    KBatch  = kernelRBF2(XBatch, params, model, "training")
    weightsBatch = compute_kpls(KBatch, (YBatch .- mean(YBatch)), model[:dim])
 
    fullNorm    = weightsBatch' * KBatch * weightsBatch

    for i = 1:model[:nsamp]

        YSam    = YBatch[Int.(model[:idxSamp][:,i]), :]
        XSam    = XBatch[Int.(model[:idxSamp][:,i]), :]

        KSam    = kernelRBF2(XSam, params, model, "training")
        KCross  = KBatch[:, Int.(model[:idxSamp][:,i])]

        weightsSam = compute_kpls(KSam, (YSam .- mean(YSam)), model[:dim])

        #println(any(isnan.(KSam)))
        #println(any(isnan.(weightsSam)))
        sampleNorm  = weightsSam' * KSam * weightsSam
        
        crossNorm   = weightsBatch' * KCross * weightsSam
        #isnan(KCross)
        lossIter    = 1. + Float64.((sampleNorm[1] - 2.0 * crossNorm[1]) / fullNorm[1])
        
        #println(any(isnan.(lossIter)))
        loss        = hcat(loss,lossIter)

    end

    lossA = sum(loss)/model[:nsamp]

    return lossA
end

function grad_rho(model::Dict{Symbol, Any})
    params  = model[:initialParams]
    beta    = 0.9
    beta2   = 0.5

    model[:np] = size(params,1)
    
    parameterHistory= zeros(model[:iter], model[:np])
    gradHistory     = zeros(model[:iter], model[:np])
    lossHistory     = zeros(model[:iter], 1)

    for i = 1:model[:iter]

        #if i == 1 || mode(i%10) == 0
            rowData = size(model[:X], 1)
            model[:idxBatch] = sort(sample(rowData, model[:sp]))
            model[:idxSamp] = zeros(ceil(Int, length(model[:idxBatch])*model[:sp]), 0)
        #end

        for i = 1: model[:nsamp]
            model[:idxSamp] = hcat(model[:idxSamp], sort(sample(length(model[:idxBatch]), model[:sp])))
        end

        for j = 1 : model[:np] 
           parameterHistory[i,j] = params[j]
           
        end

        loss, grad = withgradient(param -> rho_average(exp.(param), model[:X], model[:Y], model), params)
        #println(grad)
        lossHistory[i] = loss
        
        for j = 1:model[:np]
            if i > 1
                gradHistory[i,j]      = grad[1][j] 
            else
                gradHistory[i,j]      = 0.0
            end
            if i == 1
                params[j] = params[j] - model[:learnRate] * grad[1][j]
                
            else
                params[j] = params[j] - model[:learnRate] * (grad[1][j] + beta * (params[j] - parameterHistory[i-1,j]))
            end
        end

    end

    return model, parameterHistory, lossHistory, gradHistory
end

function optimize_parameters(model::Dict{Symbol, Any})
    
    model, parameterHistory, lossHistory, gradHistory = grad_rho(model)

    model[:runningLoss] = movmean(lossHistory, 10)
    _, model[:bestLoss] = findmin(model[:runningLoss])
    model[:bestParam] = zeros(model[:np], 1)

    for i = 1:model[:np]
        model[:bestParam][i] = parameterHistory[model[:bestLoss], i]
    end
    return model, parameterHistory, lossHistory, gradHistory
end

function movmean(x, m) 
    # this comes from github.com/chmendoza
    """Compute off-line moving average
    
    Parameters
    ----------
    x (Array):
        One-dimensional array with time-series. size(x)=(N,)
    m (int):
        Length of the window where the mean is computed
    
    Returns
    -------
    y (Array):
        Array with moving mean values. size(y)=(N-m+1)    
    """

    N = length(x)
    y = fill(0., N-m+1)

    y[1] = sum(x[1:m])/m
    for i in 2:N-m+1
        @views y[i] = y[i-1] + (x[i+m-1]-x[i-1])/m
    end
    return y
end

function predict_KF(model)
    # create the kernel Matrix for the training
        K     = kernelRBF2(model[:X], exp.(model[:bestParam]), model, "training")
        model[:K] = K
    # create the kernel Matrix for the test
        KTest = kernelRBFTest(model[:X], model[:Xtest], exp.(model[:bestParam]), model)
    #  get the regression coefficients
        B  = compute_kpls(K, (model[:Y] .- mean(model[:Y])), model[:dim])
        model[:YPred] = KTest * B
        
    return model
end
