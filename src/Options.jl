module Options
Base.@kwdef mutable struct HyperParams
    batch_size::Int = 3
    epochs::Int = 200
    lr::Float32 = 0.0002
    beta1::Float32 = 0.5
    beta2::Float32 = 0.999
    λ::Float32 = 100.0
    use_gpu::Bool = true
end
end
