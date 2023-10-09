
using DrWatson
@quickactivate "HierarchicalBayesZoo"

using DataFrames
using DataFramesMeta
using JLD2
using Plots, StatsPlots
using Statistics
using HDF5
using ProgressMeter
using SimpleUnPack

function load_data(path)
    fnames = readdir(path, join=true)
    fnames = filter(fname -> occursin("jld2", fname), fnames)
    @info("found $(length(fnames)) JLD2 files")
    dfs    = map(fnames) do fname
	    JLD2.load(fname, "data")
    end
    @info("loaded dataframes")
    vcat(dfs...)
end

function statistics(df, group_key)
    df = @chain groupby(df, group_key) begin
        @combine(:elbo_mean   = mean(:elbo),
                 :elbo_median = median(:elbo),
		 :elbo_min    = minimum(:elbo),
		 :elbo_max    = maximum(:elbo),
		 :elbo_90     = quantile(:elbo, 0.9),
		 :elbo_10     = quantile(:elbo, 0.1),
		 )
    end
end

function stepsize_slice(df, logstepsize; 
			taskname,
                        proportion,
			familyname)
    @chain df begin
        @subset(:taskname        .== taskname,
                :proportion      .== proportion,
                :familyname      .== familyname,
                :logstepsize     .== logstepsize,
                .!ismissing.(:elbo),
		isfinite.(:elbo))
        @select(:elbo, :t)
    end
end

function plot_losscurve(df, logstepsize; 
                        taskname,
                        proportion, 
                        familyname)
    df = stepsize_slice(df, logstepsize;
                        taskname   = taskname,
                        proportion = proportion,
                        familyname = familyname)

    x = df[:,:t]    |> Array{Float64}
    y = df[:,:elbo] |> Array{Float64}

    df_stats = statistics(df, :t)
    x   = df_stats[:,:t]           |> Array{Int}
    y   = df_stats[:,:elbo_median] |> Array{Float64}
    y_p = abs.(df_stats[:,:elbo_90] - y)
    y_m = abs.(df_stats[:,:elbo_10] - y)
    #display(Plots.plot!(x, y, xscale=:log10, ribbon=(y_m, y_p)))
    x, y, y_p, y_m
end

function export_losscurve(df, io=nothing; 
		          logstepsize_range,
                          taskname,
                          proportion)
    configs = [(familyname = :structured,),
               (familyname = :fullrank,),
               (familyname = :meanfield,)]
    for logstepsize in logstepsize_range
        for config in configs
            SimpleUnPack.@unpack familyname = config
            try
                x, y, y_p_abs, y_m_abs = plot_losscurve(
                    df, logstepsize;
                    taskname    = taskname,
                    proportion  = proportion,
                    familyname  = familyname)
                if !isnothing(io)
                    name = "$(string(familyname))_$(string(logstepsize))"
                    write(io, name*"_x", x)
                    write(io, name*"_y", hcat(y, y_p_abs, y_m_abs)' |> Array)
                end
            catch e
                @info("", taskname, proportion, familyname, logstepsize)
                throw(e)
            end
        end
    end
end

function export_losscurves(df = load_data(datadir("experiment")))
    configs = [
        (taskname=:irt,        proportion=0.005, logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:poisson,    proportion=0.1,  logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:volatility, proportion=0.1,  logstepsize_range=[-4, -3.5, -3]), 

        (taskname=:irt,        proportion=0.01, logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:poisson,    proportion=0.2, logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:volatility, proportion=0.2, logstepsize_range=[-4, -3.5, -3]), 

        (taskname=:irt,        proportion=0.05, logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:poisson,    proportion=1.0, logstepsize_range=[-4, -3.5, -3]), 
        (taskname=:volatility, proportion=1.0, logstepsize_range=[-4, -3.5, -3]), 
    ]

    @showprogress for config in configs
        SimpleUnPack.@unpack taskname, proportion, logstepsize_range = config
        h5open(datadir("exp_pro", "losscurve_"*savename(config)*".h5"), "w") do io
            export_losscurve(df, io; 
		             logstepsize_range,
                             taskname,
                             proportion)
        end
    end
end
