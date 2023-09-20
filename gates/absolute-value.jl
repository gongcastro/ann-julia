using Plots, Flux
using ProgressMeter

# training data
X = Matrix{Float32}(rand([-1.0, +1.0], (100, 1)))'
y = Float32.(sign.(X) .== 1)' # are BOTH channels on?

# 2 input neurons (one for each channel), 1 output neuron
# sigmoid activation
# this model will fail, because one more layer is required
abs_model_wrong = Chain(
    Dense(1, 1, sigmoid)
)

abs_model = Chain(
    Dense(1, 2, sigmoid),
    Dense(2, 1, sigmoid)
)

# initial parameters (untrained model)
params_init = Flux.params(abs_model);
params_init[1] # input-hidden weights
params_init[2]

# loss function: mean squared error (MSE)
loss_fn_wrong(a, b) = Flux.mse(abs_model_wrong(a), b)
loss_fn(a, b) = Flux.mse(abs_model(a), b)

# model setup
η = 0.001 # learning rate
opt = ADAM(η) # ADAM optimizer
N = 5_000 # number of epochs
l = zeros(N) # pre-allocate vector of losses
opt_state = Flux.setup(opt, abs_model)

# train the model
@showprogress for i in 1:N
    Flux.train!(loss_fn_wrong, Flux.params(abs_model_wrong), [(X, y')], opt)
    Flux.train!(loss_fn, Flux.params(abs_model), [(X, y')], opt)
    l[i] = loss_fn(X, y')
end

# model outputs
loss_fn(X, y') # final loss value
params_trained = params(abs_model)
params_trained[1]
params_trained[2]

# wrong model
X_test = Matrix{Float32}(rand([-1.0, +1.0], (100, 1)))'
y_test = Float32.(sign.(X_test) .== 1)' # are BOTH channels on?
preds_wrong = abs_model_wrong(X_test)
preds = abs_model(X_test)

# test accuracy is very bad
accuracy_wrong = mean(Matrix{Float32}(preds_wrong' .> 0.5) .== y_test)
accuracy = mean(Matrix{Float32}(preds' .> 0.5) .== y_test)

# plot loss
loss_plt = plot(1:N, l, xlabel="Epochs", legend=:none, ylabel="Loss (mse)")
