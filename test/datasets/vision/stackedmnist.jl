n_features = (28, 28, 3)
n_targets = 1

@testset "trainset" begin
    d = StackedMNIST(:train)
    @test d.split == :train

    @test extrema(d.features) == (0.0, 1.0)

    @test convert2image(d, 1) isa AbstractArray{<:Gray, 3}
    @test convert2image(d, 1:2) isa AbstractArray{<:Gray, 4} 

    test_supervised_array_dataset(d;
                                  n_features, n_targets, n_obs = 60000,
                                  Tx = Float32, Ty = Int,
                                  conv2img = true)

    d = StackedMNIST(:train)
    @test d.split == :train
    d = StackedMNIST(Float32, :train)
    @test d.split == :train
    @test d.features isa Array{Float32}
end

@testset "testset" begin
    d = StackedMNIST(split = :test, Tx = UInt8)

    @test d.split == :test

    @test extrema(d.features) == (0, 255)

    @test convert2image(d, 1) isa AbstractArray{<:Gray, 3} 

    test_supervised_array_dataset(d;
                                  n_features, n_targets, n_obs = 10000,
                                  Tx = UInt8, Ty = Int,
                                  conv2img = true)

    d = StackedMNIST(:test)
    @test d.split == :test
    d = StackedMNIST(Int, :test)
    @test d.split == :test
    @test d.features isa Array{Int}
end
