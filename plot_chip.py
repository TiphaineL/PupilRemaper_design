import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


points = np.load('Bezier_points.npy')

n = len(points)

t = np.arange(0,1,1e-3)

def Bezier(t,arg):
    '''
    t the dummy variable over which the Bezier curve is drawn
    arg are Bezier points, any number of points can be entered

    return a Bezier curve
    '''
    n = len(arg) - 1
    B_n = 0
    for i in range(len(arg)):
        B_t = binomial(n,i) * (1-t)**(n-i) * t**(i) * arg[i]
        B_n += B_t
    return B_n

def binomial(n,k):
    '''
    n and i integers

    return binomial(n,i)
    '''
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
for i in range(n):
    ax.plot(Bezier(t, points[i][:,0])/1000, \
            Bezier(t, points[i][:,2])/1000 ,\
            Bezier(t, points[i][:,1])/1000)
ax.set_zlim3d(-2, 2)
ax.grid(False)
ax.set_xlabel('X (mm)')
ax.set_ylabel('Z (mm)')
ax.set_zlabel('Y (mm)')
plt.show()