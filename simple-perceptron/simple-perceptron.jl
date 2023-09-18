# from Hands-On Machine Learning with Scikit-Learn and TensorFlow (Chapter 10)
# following https://fluxml.ai/Flux.jl/stable/tutorials/2021-01-26-mlp/

using Flux, Plots, MLDatasets
using DataFrames, Statistics
import Random, ProgressMeter

iris = MLDatasets.Iris()

# set model default values
Base.@kwdef mutable struct Args
    rate::Float64 = 0.001   # learning rate
    batchsize::Int = 1    # batch size
    epochs::Int = 3_000     # number of epochs
    device::Function = gpu  # set as gpu, if gpu available
end

Random.seed!(1234)

# function to get and prepare data
begin
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    iris = MLDatasets.Iris()

    # shuffle rows
    rng = MersenneTwister(1234)
    shuffled = shuffle(rng, Vector(1:nrow(iris.features)))
    X = iris.features[shuffled, :]
    y = iris.targets[shuffled, :]

    train_test_ratio = 0.70
    idx = Int(floor(size(data, 1) * train_test_ratio))

    get_feat(d) = transpose(Matrix(d))
    x_train = get_feat(X[1:idx, :])
    x_test = get_feat(X[1+idx:end, :])

    onehot(d) = Flux.onehotbatch(d[:, end], unique(d.class))
    y_train = onehot(y[1:idx, :])
    y_test = onehot(y[1+idx:end, :])

end

# create data loaders
begin
    batch_size = 1
    train_dl = Flux.Data.DataLoader((x_train, y_train), batchsize=batch_size, shuffle=true)
    test_dl = Flux.Data.DataLoader((x_test, y_test), batchsize=batch_size)
end

# define model
function build_model(; nclasses=length(unique(iris.targets[:, 1])))
    return Chain(
        Dense(4, 8, relu),
        Dense(8, nclasses),
        softmax
    )
end

model = build_model()

# define loss function
function loss_all(dataloader, model)
    l = 0.0f0
    for (x, y) in dataloader
        l += Flux.logitbinarycrossentropy(model(x), y)
    end
    l / length(dataloader)
end

# model training
function train_model(model, train_dl)
    args = Args()
    model = args.device(model)
    train_data = args.device.(train_dl)
    loss(x, y) = Flux.logitcrossentropy(m(x), y)
    ## Training
    evalcb = () -> @show(loss_all(train_data, model))
    opt = Adam(args.rate)

    ProgressMeter.@showprogress for epoch in 1:args.epochs
        Flux.train!(loss, Flux.params(model), train_data, opt)
    end

    return model

end

train_model(train_dl)

function accuracy(model, x, y)
    return mean(Flux.onecold(model(x)) .== Flux.onecold(y))
end

function confusion_matrix(model, X, y)
    ŷ = Flux.onehotbatch(Flux.onecold(model(X)), 1:3)
    y * transpose(ŷ)
end

function test(model, x_test, y_test)
    ## Testing model performance on test data 
    accuracy_score = accuracy(model, x_test, y_test)

    println("\nAccuracy: $accuracy_score")

    ## To avoid confusion, here is the definition of a 
    ## Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(model, x_test, y_test))
end

test(model, x_test, y_test)