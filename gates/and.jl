using Plots, Flux
using ProgressMeter

# training data
X = Matrix{Float32}(rand([0.0f0, 1], (100, 2)))'
y = Float32.(sum(X, dims=1) .== 2) # are BOTH channels on?

# 2 input neurons (one for each channel), 1 output neuron
# sigmoid activation
and_model = Chain(
    Dense(2, 1, sigmoid)
)

# initial parameters (untrained model)
params_init = params(and_model);
params_init[1] # input-hidden weights
params_init[2] # hidden bias

# loss function: mean squared error (MSE)
loss_fn(a, b) = Flux.mse(and_model(a), b)

# model setup
η = 0.001 # learning rate
opt = ADAM(η) # ADAM optimizer
N = 3_000 # number of epochs
l = zeros(N) # pre-allocate vector of losses
opt_state = Flux.setup(opt, model)

# train the model
@showprogress for i in 1:N
    Flux.train!(loss_fn, params(and_model), [(X, y)], opt)
    l[i] = loss_fn(X, y)
end

# model outputs
loss_fn(X, y) # final loss value
params_trained = params(model)
params_trained[1]
params_trained[2]

# test_data
X_test = Matrix{Float32}(rand([0.0f0, 1], (100, 2)))'
y_test = Float32.(sum(X_test, dims=1) .== 2) # are BOTH channels on?
preds = and_model(X_test)
# test accuracy is perfect
accuracy = mean(Matrix{Float32}(preds .> 0.5) .== y_test)

# plot loss
loss_plt = plot(1:N, l, xlabel="Epochs", legend=:none, ylabel="Loss (mse)")
