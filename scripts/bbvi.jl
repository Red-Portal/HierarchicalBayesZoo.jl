
using DrWatson
@quickactivate "HierarchicalBayesZoo"

include(srcdir("HierarchicalBayesZoo.jl"))

using .HierarchicalBayesZoo

struct BlockGaussians
    μs::Matrix
    Σs::Array
    block_idx::Vector
end

function LogDensityProblems.logdensity(model::NormalLogNormal, θ)
    @unpack μ_x, σ_x, μ_y, Σ_y = model
    logpdf(LogNormal(μ_x, σ_x), θ[1]) + logpdf(MvNormal(μ_y, Σ_y), θ[2:end])
end

function LogDensityProblems.dimension(model::NormalLogNormal)
    length(model.μ_y) + 1
end

function LogDensityProblems.capabilities(::Type{<:NormalLogNormal})
    LogDensityProblems.LogDensityOrder{0}()
end

function main()
    use_cuda     = true
    data_indices = 1:100

    update_batch = (prob, batch) -> prob
    amortize     = (   q, batch) -> q

    batchsize = 10
    n_samples = 20
    
    obj = DoublyADVI(prob,
                     n_samples;
                     batchsize    = batchsize,
                     update_batch = update_batch,
                     amortize     = amortize,
                     data_indices = data_indices,
                     use_cuda     = true)
end
