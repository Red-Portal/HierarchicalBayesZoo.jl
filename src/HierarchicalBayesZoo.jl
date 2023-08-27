
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
using LinearAlgebra: AbstractTriangular

amortize(q, x) = q

export
    ADVICUDA,
    amortize,
    Subsampling,
    IsoStructuredLocationScale,
    StructuredLocationScale

include("bbvi/advicuda.jl")
include("bbvi/subsample.jl")
include("bbvi/locscale.jl")
include("bbvi/structured.jl")

end
