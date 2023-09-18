################################################################################
# ML > Supervised Learning > Artificial Neural Network
################################################################################

# load packages

using Flux, Images, MLDatasets, Plots

using Flux: crossentropy, onecold, onehotbatch, params, train!

using Random, Statistics

# set random seed

Random.seed!(1)

# load data

X_train_raw, y_train_raw = MLDatasets.MNIST(:train)[:]

X_test_raw, y_test_raw = MLDatasets.MNIST(:test)[:]

# view training input

X_train_raw

index = 1

img = X_train_raw[:, :, index]

colorview(Gray, img')

# view training label

y_train_raw

y_train_raw[index]

# view testing input

X_test_raw

img = X_test_raw[:, :, index]

colorview(Gray, img')

# view testing label

y_test_raw

y_test_raw[index]

# flatten input data

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)

# one-hot encode labels

y_train = onehotbatch(y_train_raw, 0:9)

y_test = onehotbatch(y_test_raw, 0:9)

# define model architecture

model = Chain(
    Dense(28 * 28, 32, relu), # activation function for the second Dense layer
    Dense(32, 10),
    softmax # output activation function
)

# define loss function

loss(x, y) = crossentropy(model(x), y)

# track parameters
# this object of type parameters contains 4 arrays with the initialise parameters in the model
# the first element contains the weights of the first Dense layer
# the second contains the biases of the first Dense layer (initialised at 0)
# the third and fourth contain the same info for the second Dense layer
# these parameters will be updated according to the loss function (see above)
ps = params(model)

# optimizer
# there are other optimisers like Descent
# ADAM: version of the stochastic gradient descent (more efficient)
# decays the learning rate during learning (faster convergence)
learning_rate = Float32(0.01)
opt = ADAM(learning_rate)

# train model
loss_history = []
epochs = 500
for epoch in 1:epochs
    # train model
    train!(loss, ps, [(X_train, y_train)], opt)
    # print report
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training Loss = $train_loss")
end

# make predictions

y_hat_raw = model(X_test)

y_hat = onecold(y_hat_raw) .- 1

y = y_test_raw

mean(y_hat .== y)

# display results

check = [y_hat[i] == y[i] for i in eachindex(y)]

index = collect(1:length(y))

check_display = [index y_hat y check]

vscodedisplay(check_display)

# view misclassifications

misclass_index = 9

img = X_test_raw[:, :, misclass_index]

colorview(Gray, img')

y[misclass_index]

y_hat[misclass_index]

# initialize plot

gr(size=(600, 600))

# plot learning curve

p_l_curve = plot(1:epochs, loss_history,
    xlabel="Epochs",
    ylabel="Loss",
    title="Learning Curve",
    legend=false,
    color=:blue,
    linewidth=2
)

# save plot

savefig(p_l_curve, "ann_learning_curve.svg")