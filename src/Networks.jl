module Networks
using Flux

struct Discriminator
    model::Chain
end

function Discriminator(input_nc::Int64; ndf=64, n_layers=3, norm_layer=BatchNorm)
    use_bias = (norm_layer == InstanceNorm)
    kw = 4
    padw = 1
    model = []
    sequence = Chain(
        Conv((kw, kw), input_nc => ndf, stride=2, pad=padw),
        x -> leakyrelu(x, 0.2)
    )
    push!(model, sequence)
    nf_mult = 1
    nf_mult_prev = 1
    for n in 1:n_layers-1
        nf_mult_prev = nf_mult
        nf_mult = min(2^n, 8)
        push!(model, Chain(
            Conv((kw, kw), ndf * nf_mult_prev => ndf * nf_mult, stride=2, pad=1),
            norm_layer(ndf * nf_mult),
            x -> leakyrelu(x, 0.2)
        ))
    end
    nf_mult_prev = nf_mult
    nf_mult = min(2^n_layers, 8)
    push!(model,
        Chain(Conv((kw, kw), ndf * nf_mult_prev => ndf * nf_mult, stride=1, pad=1, bias=use_bias),
            norm_layer(ndf * nf_mult),
            x -> leakyrelu(x, 0.2))
    )
    push!(model, Chain(Conv((kw, kw), ndf * nf_mult => 1, stride=1, pad=padw)))
    model = Chain(model...)
    Discriminator(model)
end
Flux.@layer :expand Discriminator
function (d::Discriminator)(x)
    d.model(x)
end
end