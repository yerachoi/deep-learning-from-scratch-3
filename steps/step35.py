if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

# plot graph (1st)
gx = x.grad
gx.name = 'gx' + str(1)
plot_dot_graph(gx, verbose=False, to_file='tanh_diff_1.png')

# plot graph (2nd - 7th)
# (run w/ caution as it takes long since about 6th)
# iters = 6

# for i in range(iters):
#     gx = x.grad
#     x.cleargrad()
#     gx.backward(create_graph=True)

#     # plot graph
#     gx = x.grad
#     gx.name = 'gx' + str(i+2)
#     plot_dot_graph(gx, verbose=False, to_file='tanh_diff_{}.png'.format(i+2))
    