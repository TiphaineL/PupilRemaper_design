import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import datetime
import scipy.optimize
from heapq import nsmallest

plt.close('all')

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
# plt.rc('font', **font)

bend = np.arange(1000,300000,7475.0)
prox = np.arange(0,100,2.2)
stand = np.arange(0,10,.25)

bend_lim = [3000,100000]
prox_lim = [300,700]
stand_lim = [1,0.1]

F_lim = [1,0.1]

def exp_coeff(y, x):
    a = (np.log(1. / y[1]) - np.log(1. / y[0])) / (x[1] - x[0])
    b = np.log(1. / y[0]) - a * x[0]
    return np.array([a, b])


def quad_coeff(y, x):
    a = (y[1] - y[0]) / (x[1] ** 2 - x[0] ** 2)
    b = y[0] - a * x[0] ** 2
    return np.array([a, b])

bend_coeff = exp_coeff(F_lim, bend_lim)
prox_coeff = exp_coeff(F_lim, prox_lim)
stand_coeff = quad_coeff(F_lim, stand_lim)

f_bend = 5.8 / np.exp(bend_coeff[0] * bend + bend_coeff[1])
f_prox = 2.0 / np.exp(prox_coeff[0] * prox + prox_coeff[1])
f_stand = 0.3 * (stand_coeff[0] * stand ** 2 + stand_coeff[1])

# fig1 = plt.figure(1)
# plt.plot(bend,f_bend)
# plt.xlabel('Bend radius')
# plt.ylabel('Cost')
# plt.show()
#
# fig2 = plt.figure(2)
# plt.plot(prox,f_prox)
# plt.xlabel('Proximity')
# plt.ylabel('Cost')
# plt.show()
#
# fig3 = plt.figure(3)
# plt.plot(stand,f_stand)
# plt.xlabel('Pathlength std')
# plt.ylabel('Cost')
# plt.show()

# cost = np.empty((len(bend),len(prox)))
# for i in range(len(bend)):
#     for j in range(len(prox)):
#         f = 5.8 / np.exp(bend_coeff[0] * bend[i] + bend_coeff[1]) + \
#             2.0 / np.exp(prox_coeff[0] * prox[j] + prox_coeff[1])
#         cost[i,j] = f

def cost(x,y,z):
    f = 1 / np.exp(bend_coeff[0] * x + bend_coeff[1]) + \
        1 / np.exp(prox_coeff[0] * y + prox_coeff[1]) + \
        1 * (stand_coeff[0] * z ** 2 + stand_coeff[1])
    return f

# cost = np.empty( ( len(bend)) )
# for i in range(len(bend)):
#     for j in range(len(prox)):
#         for k in range(len(stand)):
#             f = 5.8 / np.exp(bend_coeff[0] * bend[i] + bend_coeff[1]) + \
#                 2.0 / np.exp(prox_coeff[0] * prox[j] + prox_coeff[1]) + \
#                 0.3 * (stand_coeff[0] * stand[k] ** 2 + stand_coeff[1])
#             cost[k] = f


X, Y, Z = np.meshgrid(bend, prox, stand)
cube = cost(X,Y,Z)
# Create the figure
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')

# Plot the values
scat = ax.scatter(X/1000, Y, Z, c = cube.ravel(), marker='o')
ax.set_xlabel('Bend radius (mm)', fontsize=14)
ax.set_ylabel('Proximity ($\mu$m)', fontsize=14)
ax.set_zlabel('Pathlength standard deviation ($\mu$m)', fontsize=14)

clb = fig.colorbar(scat, shrink=0.85)
clb.set_label('Merit function',fontsize=14, labelpad=-20, y=1.05, rotation=0)

plt.show()