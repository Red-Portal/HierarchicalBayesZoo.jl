
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

function statistics(df, group_key, statistic = :elbo)
    df = @chain groupby(df, group_key) begin
        @combine(
            $"$(statistic)_mean"   = mean($statistic),
            $"$(statistic)_median" = median($statistic),
	    $"$(statistic)_min"    = minimum($statistic),
	    $"$(statistic)_max"    = maximum($statistic),
	    $"$(statistic)_90"     = quantile($statistic, 0.9),
	    $"$(statistic)_10"     = quantile($statistic, 0.1),
	)
    end
end

function stepsize_slice(
    df, logstepsize; 
    taskname,
    proportion,
    familyname,
    statistic = :elbo
)
    @chain df begin
        @subset(:taskname        .== taskname,
                :proportion      .== proportion,
                :familyname      .== familyname,
                :logstepsize     .== logstepsize,
                .!ismissing.($statistic),
		isfinite.($statistic))
        @select($statistic, :t)
    end
end

function iteration_slice(
    df, iteration;
    taskname,
    proportion,
    familyname,
    statistic = :elbo
)
    @chain df begin
        @subset(:taskname    .== taskname,
                :proportion  .== proportion,
                :familyname  .== familyname,
                :t           .== iteration,
                .!ismissing.($statistic),
		isfinite.($statistic))
	@select($statistic, :logstepsize)
    end
end

function plot_losscurve(
    df, logstepsize; 
    taskname,
    proportion, 
    familyname,
    statistic = :elbo
)
    df = stepsize_slice(
        df, logstepsize;
        taskname   = taskname,
        proportion = proportion,
        familyname = familyname,
        statistic
    )

    x = df[:,:t]        |> Array{Float64}
    y = df[:,statistic] |> Array{Float64}

    df_stats = statistics(df, :t, statistic)
    x   = df_stats[:,:t]           |> Array{Int}
    y   = df_stats[:, Symbol("$(statistic)_median")] |> Array{Float64}
    y_p = abs.(df_stats[:,Symbol("$(statistic)_90")] - y)
    y_m = abs.(df_stats[:,Symbol("$(statistic)_10")] - y)
    #display(Plots.plot!(x, y, xscale=:log10, ribbon=(y_m, y_p)))
    x, y, y_p, y_m
end

function plot_envelope(
    df, iteration;
    taskname,
    proportion,
    familyname,
    statistic = :elbo
)
    df = iteration_slice(df, iteration;
                         taskname   = taskname,
                         proportion = proportion,
                         familyname = familyname,
                         statistic)

    x = df[:,:logstepsize] |> Array{Float64}
    y = df[:,statistic]    |> Array{Float64}

    df_stats = statistics(df, :logstepsize, statistic)
    x   = 10.0.^(df_stats[:,:logstepsize])
    y   = df_stats[:,Symbol("$(statistic)_median")]
    y_p = abs.(df_stats[:,Symbol("$(statistic)_90")] - y)
    y_m = abs.(df_stats[:,Symbol("$(statistic)_10")] - y)
    #display(Plots.plot!(x, y, xscale=:log10, ylims=(quantile(y, 0.5), Inf), ribbon=(y_m, y_p)))
    x, y, y_p, y_m
end

function export_envelope(
    df, io=nothing; 
    iteration_range,
    taskname, 
    families,
    proportion,
    statistic = :elbo
)
    configs = [(familyname = family,) for family in families]
    for iteration in iteration_range
        #display(Plots.plot())
        for config in configs
            SimpleUnPack.@unpack familyname = config
            try
            x, y, y_p_abs, y_m_abs = plot_envelope(
                df, iteration;
                taskname   = taskname,
                proportion = proportion,
                familyname = familyname,
                statistic
            )
            if !isnothing(io)
                name = "$(string(familyname))_$(string(iteration))"
                write(io, name*"_x", x)
                write(io, name*"_y", hcat(y, y_p_abs, y_m_abs)' |> Array)
            end
            catch e
                @info("", taskname, proportion, familyname, iteration)
                throw(e)
            end
        end
    end
end

function export_losscurve(
    df, io=nothing; 
    logstepsize_range,
    taskname,
    families,
    proportion,
    statistic = :elbo
)
    configs = [(familyname = family,) for family in families]
    for logstepsize in logstepsize_range
        for config in configs
            SimpleUnPack.@unpack familyname = config
            try
                x, y, y_p_abs, y_m_abs = plot_losscurve(
                    df, logstepsize;
                    taskname    = taskname,
                    proportion  = proportion,
                    familyname  = familyname,
                    statistic
                )
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

function export_losscurves(df = load_data(datadir("exp_raw")), statistic = :elbo)
    configs = [
        (taskname=:irt,        proportion=0.005, logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 
        (taskname=:poisson,    proportion=0.1,   logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 
        (taskname=:volatility, proportion=0.1,   logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 

        (taskname=:irt,        proportion=0.01,  logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 
        (taskname=:poisson,    proportion=0.2,   logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 
        (taskname=:volatility, proportion=0.2,   logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield, :fullrank]), 

        (taskname=:irt,        proportion=0.05,  logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield]), 
        (taskname=:poisson,    proportion=1.0,   logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield]), 
        (taskname=:volatility, proportion=0.99,  logstepsize_range=[-4, -3.5, -3],  families=[:structured, :meanfield]), 
    ]

    @showprogress for config in configs
        SimpleUnPack.@unpack taskname, proportion, logstepsize_range, families = config
        h5open(datadir("exp_pro", "losscurve_"*savename(config)*".h5"), "w") do io
            export_losscurve(
                df, io; 
		logstepsize_range,
                taskname,
		families,
                proportion,
                statistic
            )
        end
    end
end

function export_envelopes(df = load_data(datadir("exp_raw")), statistic = :elbo)
    configs = [
	(taskname=:irt,        proportion=0.005, iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield],), 
        (taskname=:poisson,    proportion=0.1,   iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:volatility, proportion=0.1,   iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield]), 

        (taskname=:irt,        proportion=0.01,  iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:poisson,    proportion=0.2,   iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:volatility, proportion=0.2,   iteration_range=[10001, 49901], families=[:fullrank, :structured, :meanfield]), 

        (taskname=:irt,        proportion=0.05,  iteration_range=[10001, 49901], families=[:structured, :meanfield]), 
        (taskname=:poisson,    proportion=1.0,   iteration_range=[10001, 49901], families=[:structured, :meanfield]), 
        (taskname=:volatility, proportion=0.99,  iteration_range=[10001, 49901], families=[:structured, :meanfield]), 
    ]

    @showprogress for config in configs
        SimpleUnPack.@unpack taskname, proportion, families, iteration_range = config
        h5open(datadir("exp_pro", "envelope_"*savename(config)*".h5"), "w") do io
            export_envelope(
                df, io; 
		iteration_range,
                taskname,
		families,
                proportion,
                statistic
            )
        end
    end
end

function export_envelopes_loglike(df = load_data(datadir("exp_raw")))
    configs = [
	(taskname=:irt,        proportion=0.005, iteration_range=[50000], families=[:fullrank, :structured, :meanfield],), 
        (taskname=:poisson,    proportion=0.1,   iteration_range=[50000], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:volatility, proportion=0.1,   iteration_range=[50000], families=[:fullrank, :structured, :meanfield]), 

        (taskname=:irt,        proportion=0.01,  iteration_range=[50000], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:poisson,    proportion=0.2,   iteration_range=[50000], families=[:fullrank, :structured, :meanfield]), 
        (taskname=:volatility, proportion=0.2,   iteration_range=[50000], families=[:fullrank, :structured, :meanfield]), 

        #(taskname=:irt,        proportion=0.05,  iteration_range=[50000], families=[:structured, :meanfield]), 
        #(taskname=:poisson,    proportion=1.0,   iteration_range=[50000], families=[:structured, :meanfield]), 
        #(taskname=:volatility, proportion=0.99,  iteration_range=[50000], families=[:structured, :meanfield]), 
    ]

    @showprogress for config in configs
        SimpleUnPack.@unpack taskname, proportion, families, iteration_range = config
        h5open(datadir("exp_pro", "envelope_loglike_"*savename(config)*".h5"), "w") do io
            export_envelope(
                df, io; 
		iteration_range,
                taskname,
		families,
                proportion,
                statistic = :loglike
            )
        end
    end
end
