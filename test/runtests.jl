
using DrWatson, Test
@quickactivate "HierarchicalBayesZoo"

using Distributions
using Random
using Statistics
using LinearAlgebra
using Flux, CUDA
using LogDensityProblems: logdensity

include(srcdir("HierarchicalBayesZoo.jl"))

using .HierarchicalBayesZoo

include("nnmfdirexp.jl")


