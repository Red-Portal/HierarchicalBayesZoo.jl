
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

using DiffResults
using ADTypes
using AdvancedVI

export DoublyADVI


include("bbvi/advicuda.jl")

end
