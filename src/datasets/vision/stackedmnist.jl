function __init__stacked_mnist()
    DEPNAME = "StackedMNIST"
    TRAINIMAGES = "train-images-idx3-ubyte.gz"
    TRAINLABELS = "train-labels-idx1-ubyte.gz"
    TESTIMAGES = "t10k-images-idx3-ubyte.gz"
    TESTLABELS = "t10k-labels-idx1-ubyte.gz"

    register(DataDep(DEPNAME,
                     """
                     Dataset: Stacked-MNIST
                     A modified version of the MNIST dataset where each image is stacked to create 3 channels.
                     
                     This dataset has 1000 modes and is widely used for analyzing mode collapse in GANs.

                     Authors: Yann LeCun, Corinna Cortes, Christopher J.C. Burges
                     Website: http://yann.lecun.com/exdb/mnist/
                     """,
                     "https://ossci-datasets.s3.amazonaws.com/mnist/" .*
                     [TRAINIMAGES, TRAINLABELS, TESTIMAGES, TESTLABELS],
                     "0bb1d5775d852fc5bb32c76ca15a7eb4e9a3b1514a2493f7edfcf49b639d7975"
                     # post_fetch_method = DataDeps.unpack
                     ))
end
struct StackedMNIST <: SupervisedDataset
    metadata::Dict{String, Any}
    split::Symbol
    features::Array{<:Any, 3} 
    targets::Vector{Int}
end
function StackedMNIST(; split = :train, Tx = Float32, dir = nothing)
    @assert split in [:train, :test]
    
    if split === :train
        IMAGESPATH = "train-images-idx3-ubyte.gz"
        LABELSPATH = "train-labels-idx1-ubyte.gz"
    else
        IMAGESPATH = "t10k-images-idx3-ubyte.gz"
        LABELSPATH = "t10k-labels-idx1-ubyte.gz"
    end

    features_path = datafile("StackedMNIST", IMAGESPATH, dir)
    features = bytes_to_type(Tx, MNISTReader.readimages(features_path))

    features_stacked = cat(features, features, features, dims=3)

    targets_path = datafile("StackedMNIST", LABELSPATH, dir)
    targets = Vector{Int}(MNISTReader.readlabels(targets_path))

    metadata = Dict{String, Any}()
    metadata["n_observations"] = size(features_stacked)[end]
    metadata["features_path"] = features_path
    metadata["targets_path"] = targets_path

    return StackedMNIST(metadata, split, features_stacked, targets)
end
function convert2image(::Type{<:StackedMNIST}, x::AbstractArray{<:Integer})
    convert2image(StackedMNIST, reinterpret(N0f8, convert(Array{UInt8}, x)))
end

function convert2image(::Type{<:StackedMNIST}, x::AbstractArray{T, N}) where {T, N}
    @assert N == 2 || N == 3
    x = permutedims(x, (2, 1, 3:N...))
    ImageCore = ImageShow.ImageCore
    return ImageCore.colorview(ImageCore.Gray, x)
end
function Base.getproperty(::Type{StackedMNIST}, s::Symbol)
    if s === :traindata
        @warn "StackedMNIST.traindata() is deprecated, use `StackedMNIST(split=:train)[:]` instead."
        function traindata(i; dir = nothing)
            StackedMNIST(; split = :train, dir)[i]
        end
        return traindata
    elseif s === :testdata
        @warn "StackedMNIST.testdata() is deprecated, use `StackedMNIST(split=:test)[:]` instead."
        function testdata(i; dir = nothing)
            StackedMNIST(; split = :test, dir)[i]
        end
        return testdata
    else
        return getfield(StackedMNIST, s)
    end
end
