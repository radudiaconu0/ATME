module Unetdmm
using Flux, CUDA, cuDNN
using Flux: SkipConnection
using Statistics: mean, var
using OMEinsum
using NNlib

struct SinusoidalPosEmb
    dim::Int64
end

function SinusoidalposEmb(dim::Int64)
    dim1 = dim
    return SinusoidalPosEmb(dim1)
end

Flux.@layer :expand SinusoidalPosEmb

function (emb::SinusoidalPosEmb)(x)
    half_dim = emb.dim ÷ 2
    emb = log(10000) / (half_dim - 1)
    emb = exp.(range(0, half_dim - 1) * -emb)
    emb = reshape(emb, (1, size(emb, 1))) |> gpu
    x = reshape(x, (size(x, 1), 1))
    emb = x .* emb
    emb = cat(sin.(emb), cos.(emb), dims=2)
    emb = reshape(emb, (size(emb, 2), size(emb, 1)))
    return emb
end

struct UpSample
    dim::Int64
    dim_out::Int64
    chain::Chain
end

function UpSample(dim, dim_out=Nothing)
    chain = Chain(
        Upsample(:nearest, scale=(2, 2)),
        Conv((3, 3), dim => default(dim_out, dim), pad=(1, 1)),
    )
    return UpSample(dim, dim_out, chain)
end
Flux.@layer :expand UpSample

function (u::UpSample)(x)
    return u.chain(x)
end


struct DownSample
    dim::Int64
    dim_out::Int64
    chain::Chain
end

function DownSample(dim, dim_out=Nothing)
    chain = Chain(
        x -> Rearrange(x),
        Conv((1, 1), dim * 4 => default(dim_out, dim)),
    )
    return DownSample(dim, dim_out, chain)
end

Flux.@layer :expand DownSample

function (d::DownSample)(x::AbstractArray)
    return d.chain(x)
end

function Rearrange(x::AbstractArray)
    dim1::Int64 = size(x, 1) // 2
    dim2::Int64 = size(x, 2) // 2
    dim3::Int64 = size(x, 3) * 4
    x = reshape(x, dim1, dim2, dim3, size(x, 4))
    return x
end

struct Attention{T<:Conv,S<:Conv}
    scale::Float32
    heads::Int
    to_qkv::S
    to_out::T
end


function Attention(dim::Int; heads::Int=4, dim_head::Int=32)
    scale::Float32 = dim_head^-0.5
    hidden_dim = dim_head * heads
    to_qkv = Conv((1, 1), dim => hidden_dim * 3, bias=false)
    to_out = Conv((1, 1), hidden_dim => dim)

    return Attention(scale, heads, to_qkv, to_out)
end

Flux.@layer :expand Attention

function (la::Attention)(x::AbstractArray)::AbstractArray
    h, w, c, b = size(x)
    qkv = Flux.chunk(la.to_qkv(x), 3; dims=3)
    q, k, v = qkv
    q, k, v = q[:, :, :, :], k[:, :, :, :], v[:, :, :, :]
    dim1 = size(q, 1)
    dim2 = size(q, 2)
    dim3 = size(q, 3)
    dim4 = size(q, 4)

    dim::Int64 = dim3 // la.heads

    q = reshape(q, dim, dim1 * dim2, la.heads, dim4)

    k = reshape(k, dim, dim1 * dim2, la.heads, dim4)

    v = reshape(v, dim, dim1 * dim2, la.heads, dim4)

    q = q * la.scale

    q *= la.scale

    sim = ein"dihb,djhb->ijhb"(q, k)
    attn = softmax(sim, dims=1)
    out = ein"ijhb,djhb->idhb"(attn, v)
    out = reshape(out, (Int64(size(out, 1) // h), Int64(size(out, 2)), Int64(size(out, 3) * w), Int64(size(out, 4))))
    return la.to_out(out)
end


struct WeightStandardizedConv2d{T<:Conv}
    layer::T
end

Flux.@layer :expand WeightStandardizedConv2d

function (w::WeightStandardizedConv2d)(x::AbstractArray)::AbstractArray
    w_old = w.layer.weight
    w_new = Flux.normalise(w_old; dims=ntuple(identity, ndims(w_old) - 1))
    lay_new = Conv(w_new, w.layer.bias, w.layer.σ; w.layer.stride, w.layer.pad, w.layer.dilation, w.layer.groups)
    lay_new(x)
end

function WeightStandardizedConv2d(in_channels, out_channels; kernel_size=(3, 3), stride=1, pad=0, dilation=1, groups=1)
    layer = Conv(kernel_size, in_channels => out_channels, identity; stride, pad, dilation, groups)
    return WeightStandardizedConv2d(layer)
end

struct Block
    proj::WeightStandardizedConv2d
    norm::GroupNorm
end

function Block(in_channels, out_channels; groups=8)
    proj = WeightStandardizedConv2d(in_channels, out_channels; kernel_size=(3, 3), pad=1)
    norm = GroupNorm(out_channels, groups)
    return Block(proj, norm)
end

Flux.@layer :expand Block

function (b::Block)(x; scale_shift=Nothing)
    x = b.proj(x)
    x = b.norm(x)
    if scale_shift !== Nothing
        scale, shift = scale_shift
        x = x .* scale[:, :, :, :] .+ shift[:, :, :, :]
    end
    return swish.(x)
end

struct WBlock
    model::Chain
end
Flux.@layer :expand WBlock

function WBlock()::WBlock
    layers = [
        Upsample(:bilinear; size=(64, 64)),
        Block(1, 32),
        Upsample(:bilinear; size=(128, 128)),
        Block(32, 64),
        Upsample(:bilinear; size=(256, 256)),
        Block(64, 32),
        Conv((1, 1), 32 => 1)
    ]
    return WBlock(Chain(layers...))
end

function (b::WBlock)(x)
    return tanh.(b.model(x))
end

struct LayerNorm{T<:AbstractArray}
    g::T
end

function LayerNorm(dim::Int)
    g = ones(Float32, (1, 1, dim, 1))
    LayerNorm(g)
end

Flux.@layer :expand LayerNorm

function (ln::LayerNorm)(x)
    dims = 3
    eps::Float32 = 1e-5
    μ = mean(x; dims=dims)
    σ² = var(x; dims=dims)
    x_normalized = (x .- μ) .* sqrt.((eps .+ σ²))
    return x_normalized .* ln.g
end

struct ResnetBlock
    mlp::Chain
    block1::Block
    block2::Block
    res_conv::Tuple{Conv,typeof(identity)}
end
function ResnetBlock(in_channels, out_channels; time_emb_dim=Nothing, groups=8)
    if time_emb_dim != Nothing
        mlp = Chain(
            swish,
            Dense(time_emb_dim => out_channels * 2),
        )
    else
        mlp = Chain()
    end
    block1 = Block(in_channels, out_channels; groups=groups)
    block2 = Block(out_channels, out_channels; groups=groups)
    res_conv = if in_channels == out_channels
        identity
    else
        Conv((1, 1), in_channels => out_channels)
    end
    return ResnetBlock(mlp, block1, block2, res_conv)
end

Flux.@layer :expand ResnetBlock

function (r::ResnetBlock)(x::AbstractArray; time_emb=Nothing)
    scale_shift = Nothing
    if r.mlp != Chain() && time_emb != Nothing
        time_emb = r.mlp(time_emb)
        b, c = size(time_emb)
        time_emb = reshape(time_emb, (1, 1, b, c))
        scale_shift = Flux.chunk(time_emb, 2; dims=3)
    end
    h = r.block1(x; scale_shift=scale_shift)
    h = r.block2(h; scale_shift=scale_shift)
    return r.res_conv(x) + h
end

struct RandomOrLearnedSinusoidalPosEmb{T<:AbstractArray}
    weights::T
end

function RandomOrLearnedSinusoidalPosEmb(dim::Int, is_random::Bool=false; dtype=Float32)
    @assert dim % 2 == 0
    half_dim = dim ÷ 2
    weights = randn(dtype, half_dim)
    return RandomOrLearnedSinusoidalPosEmb(weights)
end

Flux.@layer :expand RandomOrLearnedSinusoidalPosEmb

function (emb::RandomOrLearnedSinusoidalPosEmb)(x::AbstractArray)
    x = reshape(x, :, 1)
    freqs = x .* reshape(emb.weights, 1, :) .* (2 * π)
    fouriered = cat(sin.(freqs), cos.(freqs), dims=2)
    fouriered = cat(x, fouriered, dims=2)
    fouriered = transpose(fouriered)
    return fouriered
end

struct PreNorm{T<:LayerNorm,F}
    norm::T
    fn::F
end

function PreNorm(dim::Int, fn)
    norm = LayerNorm(dim)
    return PreNorm(norm, fn)
end
Flux.@layer :expand PreNorm

function (pn::PreNorm)(x::AbstractArray)
    x = pn.norm(x)
    return pn.fn(x)
end


struct LinearAttention{T<:Chain,S<:Conv}
    scale::Float32
    heads::Int
    to_qkv::S
    to_out::T
end


function LinearAttention(dim::Int; heads::Int=4, dim_head::Int=32)
    scale::Float32 = dim_head^-0.5
    hidden_dim = dim_head * heads
    to_qkv = Conv((1, 1), dim => hidden_dim * 3, bias=false)
    to_out = Chain(
        Conv((1, 1), hidden_dim => dim),
        LayerNorm(dim)  # Assuming a LayerNorm implementation as earlier
    )
    return LinearAttention(scale, heads, to_qkv, to_out)
end

Flux.@layer :expand LinearAttention

function (la::LinearAttention)(x::AbstractArray)
    h, w, c, b = size(x)
    qkv = Flux.chunk(la.to_qkv(x), 3; dims=3)
    q, k, v = qkv
    q, k, v = q[:, :, :, :], k[:, :, :, :], v[:, :, :, :]
    dim1 = size(q, 1)
    dim2 = size(q, 2)
    dim3 = size(q, 3)
    dim4 = size(q, 4)

    dim::Int64 = dim3 // la.heads
    q = reshape(q, dim, dim1 * dim2, la.heads, dim4)

    k = reshape(k, dim, dim1 * dim2, la.heads, dim4)

    v = reshape(v, dim, dim1 * dim2, la.heads, dim4)

    q = softmax(q, dims=1)
    k = softmax(k, dims=2)

    q *= la.scale
    v /= (h * w)

    context = ein"dnhb,enhb->dehb"(k, v)
    # println("context: ", size(context))
    out = ein"dehb,dnhb -> enhb"(context, q)
    # println("out: ", size(out))

    out = reshape(out, h, w, :, b)
    # println("out 2: ", size(out))
    return la.to_out(out)
end

struct UNet
    init_conv::Conv
    time_mlp::Chain
    downs::Array{Chain}
    ups::Array{Chain}
    mid_block1::ResnetBlock
    mid_attn::SkipConnection
    mid_block2::ResnetBlock
    final_res_block::ResnetBlock
    final_conv::Chain
    self_condition::Bool
end

function default(val, d)
    if val != Nothing
        return val
    else
        if d isa Function
            return d()
        else
            return d
        end
    end
end


function UNet(dim::Int64; init_dim::Int64=Nothing, out_dim=Nothing, dim_mults=(1, 2, 4, 8), channels=3, self_condition=false, resnet_block_groups=8, learned_variance=false, learned_sinusoidal_cond=false, random_fourier_features=false, learned_sunusoidal_dim=616, time_dim_mult=4)
    input_channels = if self_condition
        channels + 1
    else
        channels
    end
    init_dim = default(init_dim, dim)
    init_conv = Conv((7, 7), input_channels => init_dim; stride=1, pad=3)
    dims = [
        init_dim, [m * init_dim for m in dim_mults]...
    ]
    in_out = zip(dims[1:end], dims[2:end])
    time_dim = dim * time_dim_mult
    random_or_learned_sinusoidal_pos_cond = learned_sinusoidal_cond || random_fourier_features
    if random_or_learned_sinusoidal_pos_cond
        sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sunusoidal_dim, random_fourier_features)
        fourier_dim = learned_sunusoidal_dim + 1
    else
        sinu_pos_emb = SinusoidalposEmb(dim)
        fourier_dim = dim
    end
    time_mlp = Chain(
        sinu_pos_emb,
        Dense(fourier_dim => time_dim),
        gelu,
        Dense(time_dim => time_dim),
    )
    downs = [
        begin
            dim_in_out
            is_last = i >= length(in_out)
            dim_in, dim_out = dim_in_out
            dim_in, dim_out
            Chain(
                ResnetBlock(dim_in, dim_in; time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in; time_emb_dim=time_dim, groups=resnet_block_groups),
                SkipConnection(PreNorm(dim_in, LinearAttention(dim_in)), +),
                if is_last
                    Conv((3, 3), dim_in => dim_out, pad=1)
                else
                    DownSample(dim_in, dim_out)
                end
            )
        end
        for (i, (dim_in_out)) in enumerate(in_out)
    ]

    mid_dim = dims[end]
    mid_block1 = ResnetBlock(mid_dim, mid_dim; time_emb_dim=time_dim, groups=resnet_block_groups)
    mid_attn = SkipConnection(PreNorm(mid_dim, Attention(mid_dim)), +)
    mid_block2 = ResnetBlock(mid_dim, mid_dim; time_emb_dim=time_dim, groups=resnet_block_groups)
    ups::Array{Chain} = [
        begin
            is_last = i == length(in_out)
            dim_in, dim_out = dim_in_out
            Chain(
                ResnetBlock(dim_in + dim_out, dim_out; time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in + dim_out, dim_out; time_emb_dim=time_dim, groups=resnet_block_groups),
                SkipConnection(PreNorm(dim_out, LinearAttention(dim_out)), +),
                if is_last
                    Conv((3, 3), dim_out => dim_in, pad=(1, 1))
                else
                    UpSample(dim_out, dim_in)
                end
            )
        end
        for (i, (dim_in_out)) in enumerate(reverse(collect(in_out)))
    ]
    if learned_variance
        default_out_dim = channels * 2
    else
        default_out_dim = channels * 1
    end
    out_dim = default(out_dim, default_out_dim)
    final_res_block = ResnetBlock(dim * 2, dim; time_emb_dim=time_dim, groups=resnet_block_groups)
    final_conv = Chain(
        ResnetBlock(dim, dim; time_emb_dim=time_dim, groups=resnet_block_groups),
        Conv((1, 1), dim => out_dim),
        c -> tanh.(c)
    )
    return UNet(init_conv, time_mlp, downs, ups, mid_block1, mid_attn, mid_block2, final_res_block, final_conv, self_condition)
end

using Flux


Flux.@layer :expand UNet

function (u::UNet)(x::AbstractArray, time::AbstractArray; x_self_cond::AbstractArray=Nothing)::AbstractArray
    if u.self_condition
        if x_self_cond == Nothing
            x_self_cond = zeros(Float32, size(x, 1), size(x, 2), 1, size(x, 4))
        end
        x = cat(x, x_self_cond, dims=3)
    end
    x = u.init_conv(x)
    r = identity(x)
    time = mean(time; dims=(1, 2, 3))
    time = reshape(time, (size(time, 4), 1))
    t = u.time_mlp(time)
    x = u.downs[1][1](x; time_emb=t)
    a = identity(x)
    x = u.downs[1][2](x; time_emb=t)
    x = u.downs[1][3](x)
    b = identity(x)
    x = u.downs[1][4](x)

    x = u.downs[2][1](x; time_emb=t)
    c = identity(x)
    x = u.downs[2][2](x; time_emb=t)
    x = u.downs[2][3](x)
    d = identity(x)
    x = u.downs[2][4](x)

    x = u.downs[3][1](x; time_emb=t)
    e = identity(x)
    x = u.downs[3][2](x; time_emb=t)
    x = u.downs[3][3](x)
    f = identity(x)
    x = u.downs[3][4](x)

    x = u.downs[4][1](x; time_emb=t)
    g = identity(x)
    x = u.downs[4][2](x; time_emb=t)
    x = u.downs[4][3](x)
    h = identity(x)
    x = u.downs[4][4](x)

    x = u.mid_block1(x; time_emb=t)
    x = u.mid_attn(x)
    x = u.mid_block2(x; time_emb=t)

    x = cat(x, h, dims=3)
    x = u.ups[1][1](x; time_emb=t)
    x = cat(x, g, dims=3)
    x = u.ups[1][2](x; time_emb=t)
    x = u.ups[1][3](x)
    x = u.ups[1][4](x)

    x = cat(x, f, dims=3)
    x = u.ups[2][1](x; time_emb=t)
    x = cat(x, e, dims=3)
    x = u.ups[2][2](x; time_emb=t)
    x = u.ups[2][3](x)
    x = u.ups[2][4](x)

    x = cat(x, d, dims=3)
    x = u.ups[3][1](x; time_emb=t)
    x = cat(x, c, dims=3)
    x = u.ups[3][2](x; time_emb=t)
    x = u.ups[3][3](x)
    x = u.ups[3][4](x)

    x = cat(x, b, dims=3)
    x = u.ups[4][1](x; time_emb=t)
    x = cat(x, a, dims=3)
    x = u.ups[4][2](x; time_emb=t)
    x = u.ups[4][3](x)
    x = u.ups[4][4](x)

    x = cat(x, r, dims=3)
    x = u.final_res_block(x; time_emb=t)
    x = u.final_conv(x)
    return x
end

end
