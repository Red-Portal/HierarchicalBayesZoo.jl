
module HierarchicalBayesZoo

export NNMFDirExp, logdensity_ref

using Bijectors
using CUDA
using ChainRulesCore
using Distributions
using Functors
using LinearAlgebra
using LogDensityProblems
using LogExpFunctions
using Random
using SimpleUnPack: @unpack
using SpecialFunctions
using StatsBase
using StatsFuns

include("utils.jl")
include("bijectors.jl")
include("nnmfdirexp.jl")
include("nnmfgamgam.jl")

using Accessors
using DiffResults
using ADTypes
using AdvancedVI

amortize(q, x) = q

export ADVICUDA, Subsampling

include("bbvi/advicuda.jl")
include("bbvi/subsample.jl")
include("bbvi/locscale.jl")

end
