module ATME
using Flux, CUDA, cuDNN
using Main.Networks: Discriminator
using Flux: gpu, mean
using Statistics: mean
using MLUtils
using Optimisers
using NNlib
using Main.Unetdmm.Unetdmm: WBlock, UNet
using Main.Discpool: DiscPool, query, insert!

function gan_loss(predictions, target_is_real; target_real_label=1.0f0, target_fake_label=0.0f0)
    target_labels = (target_is_real ? target_real_label : target_fake_label)
    loss = Flux.logitbinarycrossentropy(predictions, target_labels)
    return loss
end

function l1_loss(input, target)
    loss = Flux.mae(input, target)
    return loss
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
        GC.gc()
    end
    x = cat(x..., dims=4)
    y = cat(y..., dims=4)
    x = Float32.(x)

    y = Float32.(y)
    return x, y
end


using Optimisers


generator = UNet(64; init_dim=64, out_dim=3, dim_mults=(1, 2, 4, 8), channels=3, self_condition=false, resnet_block_groups=8, learned_variance=false, learned_sinusoidal_cond=false, random_fourier_features=false, learned_sunusoidal_dim=616, time_dim_mult=4) |> gpu
discriminator = Discriminator(6; ndf=64, n_layers=3, norm_layer=BatchNorm) |> gpu
w = WBlock() |> gpu

optimizer_gen = Optimisers.Adam(0.0002, (0.5, 0.999))
optimizer_disc = Optimisers.Adam(0.0002, (0.5, 0.999))

opt_state_gen = Optimisers.setup(optimizer_gen, (w, generator))
opt_state_disc = Optimisers.setup(optimizer_disc, discriminator)
# x_train, y_train = load_data("train")
x_train, y_train = CUDA.rand(Float32, 256, 256, 3, 405), CUDA.rand(Float32, 256, 256, 3, 405)

dataloader = DataLoader((x_train, y_train), batchsize=3, shuffle=true)
discpool = DiscPool(dataloader)

function loss_discriminator(discriminator, real_A, real_B, fake_B)::Float32
    fake_AB = cat(real_A, fake_B, dims=3)
    pred_fake = discriminator(fake_AB)
    loss_D_fake = gan_loss(pred_fake, false)
    real_AB = cat(real_A, real_B, dims=3)
    pred_real = discriminator(real_AB)
    loss_D_real = gan_loss(pred_real, true)
    loss_D = (loss_D_fake + loss_D_real) * 0.5f0
    return loss_D
end

function loss_generator(w, gen, discriminator, disc_B, real_A, real_B)
    Disc_B = w(disc_B)

    noisy_A = real_A .* (1.0f0 .+ Disc_B)
    fake_B = gen(noisy_A, Disc_B)

    fake_AB = cat(real_A, fake_B, dims=3)
    disc_B = discriminator(fake_AB)
    loss_G_GAN = gan_loss(disc_B, true)
    loss_G_L1 = l1_loss(fake_B, real_B) * 100.0f0
    loss_G = loss_G_GAN + loss_G_L1
    return (loss_G, loss_G_GAN, loss_G_L1, disc_B)
end

function save_images(dir, epoch, real_A, real_B, fake_B, noisy_A)
    img = sigmoid.(cpu(fake_B[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_fakeB.png"), img)

    img = sigmoid.(cpu(real_B[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_realB.png"), img)

    img = sigmoid.(cpu(real_A[:, :, :, 2]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_realA.png"), img)

    img = sigmoid.(cpu(noisy_A[:, :, :, 1]))
    img = colorview(RGB, permutedims(img, (3, 2, 1)))
    Images.save(joinpath(dir, "img_epoch_$(epoch)_noisyA.png"), img)
end
using Optimisers
using ProgressBars

loss_D = 0.0
loss_G = 0.0
loss_G_GAN = 0.0
loss_G_L1 = 0.0
GC.gc()

function train_discriminator!(discriminator, w, generator, real_A, real_B, disc_B)
    Disc_B = w(disc_B)
    noisy_A = real_A .* (1.0f0 .+ Disc_B)
    fake_B = generator(noisy_A, Disc_B)

    loss_D, grads_D = Flux.withgradient(discriminator) do disc
        loss_D = loss_discriminator(disc, real_A, real_B, fake_B)
        loss_D
    end

    Optimisers.update!(opt_state_disc, discriminator, grads_D[1])
    return loss_D
end

function train_generator!(w, generator, discriminator, real_A, real_B, disc_B)

    loss_G, grads_G = Flux.withgradient(w, generator) do W, Gen
        loss_G, loss_G_GAN, loss_G_L1, disc_B = loss_generator(W, Gen, discriminator, disc_B, real_A, real_B)
        loss_G
    end
    Optimisers.update!(opt_state_gen, (w, generator), grads_G)
    return loss_G, loss_G_GAN, loss_G_L1, disc_B
end

using ProgressBars

query(discpool, 1:3)
for epoch in 1:20
    iter = ProgressBar(dataloader)
    println("epoch $epoch")
    GC.gc()
    start_idx = 1
    index = 0
    for (x, y) in iter
        index += 1
        batchIdx = LinRange{Int64}(start_idx, start_idx + 2, 3)
        disc_B = query(discpool, batchIdx)
        real_A = x |> gpu
        real_B = y |> gpu

        loss_D = train_discriminator!(discriminator, w, generator, real_A, real_B, disc_B)


        loss_G, loss_G_GAN, loss_G_L1, disc_B = train_generator!(w, generator, discriminator, real_A, real_B, disc_B)

        insert!(discpool, disc_B, batchIdx)
        start_idx += 3

        GC.gc()
        set_multiline_postfix(iter, "loss_D:$(loss_D)\nloss_G:$(loss_G), loss_G_GAN:$(loss_G_GAN), loss_G_L1:$(loss_G_L1)")
    end
    GC.gc()
end

end # module ATME
