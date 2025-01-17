import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from constrained_learning.regions import Box
from constrained_learning.learner import CELM
from constrained_learning.constraints import ObjFct, DEQC

# first order ODE
x = np.linspace(0, 2*np.pi)
y = np.cos(x)

# Define the model and constraints
region = Box(lower_bounds=[-3], upper_bounds=[3])
obj_fct = ObjFct(partials=[[[0]]], factors=[[.5]])  # f'(x)=2*cos(x)
con = DEQC(partials=[[[]]], factors=[[1]], value=0, u=np.array([0]))  # f(0)=0
model = CELM(inp_dim=1,
             out_dim=1,
             hid_dim=30,
             deqcs=[con],
             obj_fcts=[obj_fct])

# Fit the ode
model.init(x)
model.train(x, y)
y_hat = model.apply(x)

# Plotting
plt.figure(figsize=(10, 7))
plt.plot(x, y, color='k', label='Ground Truth')
plt.plot(x, y_hat, color='r', label="Solution for f'(x)=2*cos(x)")
plt.gca().set_title('First Order ODE with Discrete Constraint at (0,0)')
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('f(x)')
plt.legend()
plt.show()
