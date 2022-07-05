#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np

x_data = np.array([[1],[2],[3],[4],[5],[6],[7],[8]],dtype='float32')

y_data = np.array([[2],[4],[6],[8],[10],[12],[14],[16]],dtype='float32')
y_data = 5*x_data + 22
print("y_data = ", y_data)

x_data = torch.from_numpy(x_data)
y_data = torch.from_numpy(y_data)

# x_data = Variable(torch.Tensor([[1],[2],[3],[4],[5],[6],[7],[8]]))
# y_data = Variable(torch.Tensor([[2],[4],[6],[8],[10],[12],[14],[16]]))

# class LinearRegressionModel(torch.nn.Module):
    
# 	def __init__(self):
# 		super(LinearRegressionModel, self).__init__()
# 		self.linear = torch.nn.Linear(1, 1) # One in and one out

# 	def forward(self, x):
# 		y_pred = self.linear(x)
# 		return y_pred

# # our model
# our_model = LinearRegressionModel()

# criterion = torch.nn.MSELoss(size_average = False)
# optimizer = torch.optim.SGD(our_model.parameters(), lr = 0.01)
# #lr is learning rate

# for epoch in range(500):
    
# 	# Forward pass: Compute predicted y by passing
# 	# x to the model
# 	pred_y = our_model(x_data)

# 	# Compute and print loss
# 	loss = criterion(pred_y, y_data)

# 	# Zero gradients, perform a backward pass,
# 	# and update the weights.
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
# 	print('epoch {}, loss {}'.format(epoch, loss.item()))

# new_var = Variable(torch.Tensor([[4.0]]))
# pred_y = our_model(new_var)
# print("predict (after training)", 4, our_model(new_var).item())


#  method 2

from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F


# w = torch.randn(1, requires_grad=True)

train_ds = TensorDataset(x_data, y_data)

# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# Define model
model = nn.Linear(1,1)

print("Weight " ,model.weight)
print("Bias " ,model.bias)

# Generate predictions
preds = model(x_data)

# Define loss function
loss_fn = F.mse_loss

loss = loss_fn(model(x_data), y_data)
print(loss)

# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-3)

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

fit(10000, model, loss_fn, opt, train_dl)
preds = model(x_data)
print('Predictions :', preds)
print("y_data = ", y_data)

