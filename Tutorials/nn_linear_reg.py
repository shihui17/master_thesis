import torch
import numpy as np
from torch.autograd import grad
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)


n_samples, n_features = X.shape
print(n_samples, n_features)

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

# model prediction

# loss = MSE

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2 * x * (w*x - y)


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)
    #print(y_pred)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward() #dl/dw

    # update weights
    optimizer.step()
    
    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')

