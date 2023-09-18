using Flux, Statistics, LinearAlgebra
using Plots, Random, DataFrames, CSV, ProgressMeter

Random.setseed!(1234)

# preprocess data --------------------------------------------------------
bmi_train = CSV.read("bmi/bmi_train.csv", DataFrame)
bmi_test = CSV.read("bmi/bmi_train.csv", DataFrame)

feature_names = [:Gender, :Height, :Weight]
target_names = [:Index]

# recode string to Int
function recode_strings(data::DataFrame)
    str_idx = findall(col -> eltype(col) <: String7, eachcol(data))
    str_nms = propertynames(data)[str_idx]
    values = Matrix(data[:, str_idx])
    function recode(x)
        Int.(x .== unique(x)[1])
    end
    data = transform(data, str_nms => recode, renamecols=false)
    return data
end

bmi_train = recode_strings(bmi_train)
bmi_test = recode_strings(bmi_test)

# prepare data for modelling ---------------------------------------------
function simple_loader(data::DataFrame; feature_names::Vector{Symbol}, target_names::Vector{Symbol}, batchsize::Int=64)
    X = Matrix{Int32}(data[:, feature_names])'
    y = Matrix(data[:, target_names])
    yhot = Flux.onehotbatch(y, unique(y))[:, :]
    Flux.DataLoader((X, yhot); batchsize, shuffle=true)
end

function simple_accuracy(model, loader::Flux.DataLoader)
    (x, y) = only(loader)  # make one big batch
    y_hat = model(x)
    iscorrect = Flux.onecold(y_hat) .== Flux.onecold(y)  # BitVector
    acc = round(100 * mean(iscorrect); digits=2)
end

# model structure -----------------------------------------
nfeatures = length(feature_names)
ncategory = nrow(unique(bmi_train[:, target_names]))

model = Chain(
    Dense(nfeatures => 10, tanh),   # activation function inside layer
    Dense(10 => ncategory),
    softmax
)

# train model -------------------------------------------
optim = Flux.setup(Flux.Adam(0.001), model)  # will store optimiser momentum, etc.
losses = []
gradients = []
epochs = 10_000

loader_train = simple_loader(
    bmi_train,
    feature_names=feature_names,
    target_names=target_names,
    batchsize=nrow(bmi_train)
)

loader_test = simple_loader(
    bmi_test,
    feature_names=feature_names,
    target_names=target_names,
    batchsize=nrow(bmi_test)
)

x1, y1 = first(loader_train)

@showprogress for epoch in 1:epochs
    loss = 0.0
    for (x, y) in loader_train
        l, g = Flux.withgradient(m -> Flux.binarycrossentropy(m(x), y), model)
        Flux.update!(optim, model, g[1])
        loss += l / length(loader_train)
    end

    if mod(epoch, 2) == 1
        # Report on train and test, only every 2nd epoch:
        train_acc = simple_accuracy(model, loader_train)
        test_acc = simple_accuracy(model, loader_test)
        @info "After epoch = $epoch" loss train_acc test_acc
    end
end





simple_accuracy(model, loader_test)

x_test, y_test = only(loader_test)
model(x_test)
preds = Flux.onecold(y_test, collect(1:ncategory))

bmi_test.Prediction = preds
bmi_test.Correct = bmi_test.Index .== bmi_test.Prediction