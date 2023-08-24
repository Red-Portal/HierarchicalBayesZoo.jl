
@testset "nnmfdirexp" begin
    U       = 256
    I       = 512

    α       = 1.0f0
    λ₀      = 1.0f0
    K       = 5
    likeadj = 1.0f0

    β = rand(Exponential(λ₀), I, K)
    θ = rand(Dirichlet(K, λ₀), U)
    λ = β*θ
    y = convert(Matrix{Int32}, @. rand(Poisson(λ)))

    λ_β = randn(Float32, I, K)
    λ_θ = randn(Float32, K-1, U)

    λ_β_dev = Flux.gpu(λ_β)
    λ_θ_dev = Flux.gpu(λ_θ)

    model     = NNMFDirExp(α, λ₀, y, K, I, U, likeadj)
    model_dev = Flux.gpu(model)

    λ_flat = vcat(reshape(λ_β_dev,:), reshape(λ_θ_dev,:))
    @test logdensity(model_dev, λ_flat) ≈ logdensity_ref(model, λ_β, λ_θ)

    # custom_host_t = @belapsed begin
    #     logdensity($model, $λ_β, $λ_θ)
    # end
    # custom_dev_t = @belapsed begin
    #     logdensity($model_dev, $λ_β_dev, $λ_θ_dev)
    # end
    # reference_t = @belapsed begin
    #     logdensity_ref($model, $λ_β, $λ_θ)
    # end
    # custom_host_t, custom_dev_t, reference_t
end
