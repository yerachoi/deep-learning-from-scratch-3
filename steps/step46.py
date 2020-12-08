if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# create data
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

# set hyperparams
lr = 0.2
max_iter = 10000
hidden_size = 10

# # define model (optimizer: SGD)
# model = MLP((hidden_size, 1))
# optimizer = optimizers.SGD(lr)
# optimizer.setup(model)
# # equivalent one-liner
# # optimizer = optimizers.SGD(lr).setup(model)

# define model (optimizer: MomentumSGD)
model = MLP((hidden_size, 1))
optimizer = optimizers.MomentumSGD(lr=lr)
optimizer.setup(model)
# equivalent one-liner
# optimizer = optimizers.MomentumSGD(lr).setup(model)

# train
for i in range(max_iter):
    y_pred = model(x)
    loss = F.mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    optimizer.update()
    if i % 1000 == 0:
        print(loss)