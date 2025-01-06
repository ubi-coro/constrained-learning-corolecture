import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from constrained_learning.regions import Box
from constrained_learning.learner import CELM
from constrained_learning.constraints import CIEQC
from constrained_learning.utils import make_seq_from_grid, make_grid_from_seq

# quadratic
x = np.linspace(-3, 3)
y = 2*x**2 - 3

# Define the model and constraints
region = Box(lower_bounds=[-3], upper_bounds=[3])
con = CIEQC(region=region,
            max_value=np.inf,
            min_value=0,
            partials=[[[0]]],
            factors=[[1]])  # constrain first derivative
model = CELM(inp_dim=1,
             out_dim=1,
             hid_dim=30,
             callbacks=[lambda i, model: print(f"Iteration {i}!")],
             cieqcs=[con])

# Fit the quadratic
model.init(x)
model.train(x, y)
y_hat = model.apply(x)

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(x, y, color='k', label='Ground Truth')
plt.plot(x, y_hat, color='r', label="Constrained Approximation [f'(x) > 0]")

if con.u.size != 0:
    u = np.atleast_2d(con.u)
    z_u = model.apply(u)
    plt.scatter(u[:, 0], z_u, color='b', s=70, label='Constraint Samples')

plt.gca().set_title('Monotone Quadratic')
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('f(x)')
plt.legend()
plt.show()
