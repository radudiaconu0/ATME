using Main.Options: HyperParams
using Flux, CUDA, cuDNN
using Main.Networks: Discriminator
using Flux: gpu, mean
using Statistics: mean
using MLUtils
using Optimisers
using NNlib
using Main.Unetdmm.Unetdmm: WBlock, UNet
using Main.Discpool: DiscPool, query, insert!




function gan_loss(predictions, target_is_real; target_real_label=1.0f0, target_fake_label=0.0f0)::Float32
    target_labels = (target_is_real ? target_real_label : target_fake_label)
    loss = Flux.logitbinarycrossentropy(predictions, target_labels)
    return loss
end

function l1_loss(input, target)::Float32
    loss = Flux.mae(input, target)
    return loss
end

function loss_discriminator(real_A, real_B, fake_AB, discriminator::Discriminator)::Float32
    pred_fake = discriminator(fake_AB)
    loss_D_fake = gan_loss(pred_fake, false)
    real_AB = cat(real_A, real_B, dims=3)
    pred_real = D(real_AB)
    loss_D_real = gan_loss(pred_real, true)
    loss_D = (loss_D_fake + loss_D_real) * 0.5f0
    return loss_D
end


using ImageCore, Images

function load_data(split::String)::Tuple{Array{Float32,4},Array{Float32,4}}
    dir = split
    dir = joinpath(@__DIR__, dir)
    dirs = readdir(dir)
    x = []
    y = []
    dirs = dirs[1:4]
    for d in dirs
        blur = "blur"
        sharp = "sharp"
        blur = joinpath(dir, d, blur)
        sharp = joinpath(dir, d, sharp)
        blurs = readdir(blur)
        sharps = readdir(sharp)
        ln = length(blurs)
        for i in 1:ln
            b = load(joinpath(blur, blurs[i]))
            s = load(joinpath(sharp, sharps[i]))
            b = Float32.(channelview(b))
            s = Float32.(channelview(s))
            b = permutedims(b, (3, 2, 1))
            s = permutedims(s, (3, 2, 1))
            b = imresize(b, (256, 256))
            s = imresize(s, (256, 256))
            push!(x, b)
            push!(y, s)
        end
    end
    x = cat(x..., dims=4)
    y = cat(y..., dims=4)
    x = Float32.(x)

    y = Float32.(y)
    return x, y
end



function save_images(dir, epoch, real_A, real_B, fake_B, noisy_A, index)
    img = sigmoid.(cpu(fake_B[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_$(index)_fakeB.png"), img)

    img = sigmoid.(cpu(real_B[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_$(index)_realB.png"), img)

    img = sigmoid.(cpu(real_A[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_$(index)_realA.png"), img)

    img = sigmoid.(cpu(noisy_A[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_$(index)_noisyA.png"), img)
end

function train_discrimniniator!(discriminator, real_A, real_B, fake_AB, opt_disc)
    loss_D, grads_D = Flux.withgradient(discriminator) do D
        return loss_discriminator(real_A, real_B, fake_AB, D)
    end
    Flux.update!(opt_disc, discriminator, grads_D[1])
    return loss_D
end

using Optimisers

function train_GAN!(w::WBlock, generator::UNet, discriminator::Discriminator, real_A::AbstractArray, real_B::AbstractArray, disc_B::AbstractArray, opt_gen, opt_disc, hp::HyperParams)
    (loss_G, fake_B, noisy_A), (grads_W, grads_G) = Flux.withgradient(w, generator) do W, G
        Disc_B = W(disc_B)
        noisy_A = real_A .* (1.0f0 .+ Disc_B)
        fake_B = G(noisy_A, Disc_B)
        fake_AB = cat(real_A, fake_B, dims=3)
        ignore() do
            loss_D = train_discrimniniator!(discriminator, real_A, real_B, fake_AB, opt_disc)
            push!(losses_D, loss_D)
        end
        disc_B = discriminator(fake_AB)
        loss_G_GAN = gan_loss(disc_B, true)
        loss_G_L1 = l1_loss(fake_B, real_B) * hp.λ
        loss_G = loss_G_GAN + loss_G_L1
        return loss_G, fake_B, noisy_A
    end
    Flux.update!(opt_gen, (w, generator), (grads_W, grads_G))
    push!(losses_G, loss_G)
    return disc_B, fake_B, noisy_A, losses_D[end], losses_G[end]
end

function train(; kws...)
    if CUDA.functional()
        @info "Training on GPU"
    else
        @info "Training on CPU"
    end
    hp = HyperParams(; kws...)
    generator = UNet(64; init_dim=64, out_dim=3, dim_mults=(1, 2, 4, 8), channels=3, self_condition=false, resnet_block_groups=8, learned_variance=false, learned_sinusoidal_cond=false, random_fourier_features=false, learned_sunusoidal_dim=616, time_dim_mult=4) |> gpu
    discriminator = Discriminator(6; ndf=64, n_layers=3, norm_layer=BatchNorm) |> hp.use_gpu ? gpu : cpu
    w = WBlock() |> hp.use_gpu ? gpu : cpu

    optimizer_gen = Flux.setup(Adam(hp.lr, (hp.beta1, hp.beta2)), (w, generator))
    optimizer_disc = FLux.setup(Adam(hp.lr, (hp.beta1, hp.beta2)), discriminator)


    for epoch in hp.epochs
        iter = ProgressBar(dataloader)
        println("Epoch $epoch")
        start_idx = 1
        index = 0
        GC.gc()
        CUDA.reclaim()
        for (x, y) in iter
            index += batch_size
            real_A = x |> gpu
            real_B = y |> gpu
            batchIdx = LinRange{Int64}(start_idx, start_idx + hp.batch_size - 1, hp.batch_size)
            disc_B = query(discpool, batchIdx)
            disc_B, fake_B, noisy_A, loss_D, loss_G = train_GAN!(w, generator, discriminator, real_A, real_B, disc_B, optimizer_disc, optimizer_gen, hp)
            insert!(discpool, disc_B, batchIdx)
            start_idx += hp.batch_size
            if index % 135 == 0
                save_images("images", epoch, real_A, real_B, fake_B, noisy_A, index)
            end
            set_multiline_postfix(iter, "loss_D: $(loss_D)\nloss_G: $(loss_G)")
            GC.gc()
        end
    end
end

