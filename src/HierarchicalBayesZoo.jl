
module HierarchicalBayesZoo

export NNMFDirExp, Volatility, logdensity_ref
export
    ADVICUDA,
    amortize,
    Subsampling,
    AmortizedLocationScale,
    IsoStructuredLocationScale,
    StructuredLocationScale

using Accessors
using Bijectors
using CUDA
using CSV
using ChainRulesCore
using DataFrames
using DataFramesMeta
using Dates
using Distributions
using DrWatson
using Functors
using Flux
using FillArrays
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using Random
using SimpleUnPack: @unpack
using SpecialFunctions
using StatsBase
using StatsFuns
using Optimisers
using NNlib
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
 
include("utils.jl")
include("bijectors.jl")
include("volatility.jl")
include("nnmfdirexp.jl")
include("nnmfgamgam.jl")

end
