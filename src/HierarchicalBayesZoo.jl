
module HierarchicalBayesZoo

export
    problem,
    #Gaussian,
    #MovieLensNNMF,
    BSSNNMF,
    CritLangIRT,
    ForeignExchangeVolatility

export
    ADVICUDA,
    amortize,
    Subsampling,
    AmortizedLocationScale,
    StructuredGaussian

using CSV
using DelimitedFiles
using DataFrames
using DataFramesMeta

using Accessors
using Bijectors
using CUDA
using ChainRulesCore
using Dates
using Distributions
using DrWatson
using FillArrays
using Flux
using Functors
using KernelAbstractions
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using NNlib
using Optimisers
using Random
using SignalAnalysis
using SimpleUnPack: @unpack
using SparseArrays
using SpecialFunctions
using StatsBase
using StatsFuns
using Zygote: @adjoint

using DiffResults
using ADTypes
using AdvancedVI
using LinearAlgebra: AbstractTriangular

include("bbvi/advicuda.jl")
include("bbvi/subsample.jl")
include("bbvi/locscale.jl")
include("bbvi/structured.jl")

subsample_problem(prob, batch) = prob
amortize(prob, q, batch) = q
 
include("utils.jl")
include("bijectors.jl")

include("models/volatility/model.jl")
include("models/volatility/interfaces.jl")

include("models/nnmfdirexp/model.jl")
include("models/nnmfdirexp/interfaces.jl")

include("models/nnmfgamgam/model.jl")
include("models/nnmfgamgam/interfaces.jl")

include("models/irt/model.jl")
include("models/irt/interfaces.jl")

#include("gaussians.jl")

end
