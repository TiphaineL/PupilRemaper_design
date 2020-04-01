#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 11:58:20 2017

@author: tiphainelagadec
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import datetime
import scipy.optimize
from heapq import nsmallest
from heapq import nlargest
import random

plt.close('all')
t0 = datetime.datetime.now()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 30}

#############################################################################
# Parameters (all lengths in microns)

# x is the horizontal axis, y the vertical and z the depth

#L = 30793.68 # Length of the remaping chip

#MEMS_pupil = np.array( [[-0.909,2.1,0.],\
#                        [1.515,-0.7,0.],\
#                        [0.909,-2.1,0.],\
#                        [-0.909,-2.1,0.]])


# 2 waveguides
MEMS_pupil = np.array( [[2.1,-0.909,0.],\
                        [-0.7,1.515,0.],\
                        [-2.1,0.909,0.],
                        [-2.1,-0.909,0.]])

#in_put = np.array( [[-77.94,15,0],\
#                    [-51.96,60,0],\
#                    [-51.96,-60,0],\
#                    [0,90,0]] )

### 3 waveguides
#MEMS_pupil = np.array( [[2.1,-0.909,0.],\
#                        [-0.7,1.515,0.],\
#                        [-2.1,0.909,0.]])


in_put = 50 * MEMS_pupil

n = len(in_put) # number of waveguides


t = np.arange(0,1,1e-3) # t is the dummy variable that we will use in the
                        # definition of the Bezier curves

#fig = plt.figure(1)
#plt.scatter(in_put[:,0],in_put[:,1],s=300)
#plt.show()
#############################################################################
# Functions

### Vector manipulation
def zero_array(n):
    '''
    n an integer

    create a n*n array of zeros
    '''
    return np.zeros((n,n))

def point_to_vector(end,start):
    '''
    start, the start point is an array
    end, the end point is an array

    convert them into a vector
    '''
    return abs(end-start)

def vector_length(vector):
    '''
    vector a vector (in 3D)

    return the length of the vector
    '''
    return np.sqrt(vector[0]**2+vector[1]**2+vector[2]**2)

def cross_product(a,b):
    '''
    a and b two vectors (in 3D)
    
    return the cross product of axb
    '''
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]
    return np.array([x,y,z])

def notZeroMin(array):
    '''
    array an array

    returns the minimum that is not zero
    '''
    return np.amin(array[np.nonzero(array)])

def getIndex(array,a):
    '''
    array an array
    a a float

    return the index in array of the position of a
    '''
    return [np.where(array == a)[0][0],np.where(array == a)[1][0]]

### Bezier functions
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

def pts_derivative(arg):
    '''
    arg are the ctl pts of the Bezier curve
    
    returns the ctl pts of the derivatibve of the curve
    '''
    n = len(arg)
    P_i = []
    for i in range(n-1):
        P_i.append((n-1)*(arg[i+1]-arg[i]))
    return np.array(P_i)

def Bezier_derivative(t,arg):
    '''
    arg are the control Point of the Bezier curve

    returns the Bezier curve that is the derivative to the Bezier curve
    of control points arg
    '''
    return Bezier(t,pts_derivative(arg))

def pts_n_plus_1(arg):
    '''
    arg the control pts
    
    returns the control points of the Bezier curve with degree plus 1
    '''
    n = len(arg)
    P_i = []
    for i in range(n):
        P_i.append((float(n-i)/n)*arg[i] + (float(i)/n)*arg[i-1])
    P_i.append(arg[n-1])
    return np.array([P_i])
    
def Bezier_n_plus_1(t,arg):
    '''
    arg are Bezier points, any number of points can be entered

    return Bezier curve of degree n+1
    '''
    return Bezier(t,pts_n_plus_1(arg))

def arc_length(array):
    '''
    array is an array with all the control points
    each line is a vector point corresponding to a
    control point P_i
    the column correspond to the dimension (x,y,z)

    returns the arc lenght of the curve
    '''
    F_x = Bezier_derivative(t,array[:,0])
    F_y = Bezier_derivative(t,array[:,1])
    F_z = Bezier_derivative(t,array[:,2])

    F = np.array([F_x,F_y,F_z])
    L = vector_length(F)
    return sum(L)/len(L)

def Curvature(array):
    '''
    array is an array with all the control points
    each line is a vector point corresponding to a
    control point P_i
    the column correspond to the dimension (x,y,z)

    Returns the curvature of the curve as a function of t
    '''

    P = array
    P_dot     = pts_derivative(P)
    P_dot_dot = pts_derivative(P_dot)
   
    B         = np.array([Bezier(t,P[:,0]),\
                          Bezier(t,P[:,1]),\
                          Bezier(t,P[:,2])])
    
    B_dot     = np.array([Bezier(t,P_dot[:,0]),\
                          Bezier(t,P_dot[:,1]),\
                          Bezier(t,P_dot[:,2])])
    
    B_dot_dot = np.array([Bezier(t,P_dot_dot[:,0]),\
                          Bezier(t,P_dot_dot[:,1]),\
                          Bezier(t,P_dot_dot[:,2])])
    
    return vector_length(cross_product(B_dot,B_dot_dot))\
                           /(vector_length(B_dot))**3

### Function relative to the problem
def center_chip(inputPoints):
    '''
    inputPoints is an array of points in 3D

    returns the center of the points in the x direction
    '''
    x = []
    for i in range(len(inputPoints)):
        x.append(inputPoints[i][0])
    center = np.mean([max(x),min(x)])
    return center

def output(inputPoints,L):
    '''
    inputPoints is an array of points in 3D

    returns an array of output points 
    the points lie in a line on the x axis at y=0
    '''
    out_put = []
    x_offset = 4000.0
    x_offset = 10000.
    step = 254
    step = 127
    x1 = center_chip(inputPoints) + x_offset
    
    for i in range(len(inputPoints)):
        out_put.append([x1,-700.,L])
        x1 += step
    return np.array(out_put)

def all_pathLength(in_put,out_put):
    '''
    in_put the input array with all the input positions
    out_put the out_put array with all the output positions

    returns an array with all the pathlength possible for
    all input/output combination possible
    '''
    n = len(in_put)
    array = zero_array(n)
    
    for i in range(n):
        for j in range(n):
            vector = point_to_vector(out_put[j],in_put[i])
            array[i][j] = vector_length(vector)
    return array


def array_mean_dev(array):
    '''
    array an array

    returns an array with the deviation for each point from the mean
    '''
    return abs(array - np.mean(array))


def get_connections(Input,L):
    '''
    Force the connections
    ie. A goes to 1, B to 2, C to 3 and D to 4
    '''
    Output = output(Input,L)
    
#    connect = np.array([[Input[0],Output[0]],\
#                    [Input[1],Output[1]]])
    
#    connect = np.array([[Input[0],Output[0]],\
#                        [Input[1],Output[2]],\
#                        [Input[2],Output[3]],\
#                        [Input[3],Output[1]]])
#    
    connect = np.array([[Input[0],Output[1]],\
                        [Input[1],Output[3]],\
                        [Input[2],Output[2]],\
                        [Input[3],Output[0]]])
    
    return(connect)


def proximity(t,P,P_A):
    '''
    t the Bezier dummy variable
    P the control points on the first curve B1
    P_A the control points on the second curve B2
    
    calculates the minimum distance between B1 and B2 by sliding a point A on
    B2
    Returns the minimum distance as well as the corresponding z-position 
    (along propagation)
    '''
    
    # Define the cordinates x, y and z for the first curve
    x = Bezier(t,P[:,0])
    y = Bezier(t,P[:,1])
    z = Bezier(t,P[:,2])
    
    min_dist = []
    z_ax = []

    for i in range(len(t)):
        # xA, yA et zA is sliding along B2 in the for loop
        xA = Bezier(t,P_A[:,0])[i]
        yA = Bezier(t,P_A[:,1])[i]
        zA = Bezier(t,P_A[:,2])[i]
    
        # for each position i of the point A, it calculates all the 
        # distances with all the points of the other curve
        S = np.sqrt( (x-xA)**2 + (y-yA)**2 + (z-zA)**2 )
        
        # take the minimum distance
        min_dist.append(min(S))
        
        # also save the corresponding z position
        # in that way the proximity is roughly defined as a function of z
        # rather than t
        z_ax.append(zA)
        
    return np.array([z_ax,min_dist])

def proximity2(t,P,P_A):
    
    x = Bezier(t,P[:,0])
    y = Bezier(t,P[:,1])
    z = Bezier(t,P[:,2])
    
    xA = Bezier(t,P_A[:,0])
    yA = Bezier(t,P_A[:,1])
    zA = Bezier(t,P_A[:,2])

    dist = np.sqrt( (x[:,None] - xA[None,:])**2 +\
                    (y[:,None] - yA[None,:])**2 +\
                    (z[:,None] - zA[None,:])**2 )
    min_dist = dist.min(axis=0)

    return np.array([zA,min_dist])
    
def guess_points(P,x):
    '''
    P the input positions points
    x the variables (the z position for the two intermediate control points)
    
    
    '''
    # the length of the chip (variable)
    L = 30800.
    #L = 31000.
    #L = 30000.
#    print 'L =',L
    # n is the number of waveguides
    n = len(P)
    # The input position points
    P_in = P

    
    # Match input positions with output positions (how decided??)
    connections = get_connections(P_in,L)
    

    waveguides = []
   
    ### Iterating through each waveguide
    for i in range(n):
        ### Start and end points for each waveguide
        Points = np.array([connections[i][0],connections[i][1]])
        ### add two more control points in order to be able to create
        ### cubic Bezier curves
#        Points = np.array( [ Points[0],\
#                           [Points[0][0],Points[0][1],x[2*i]],\
#                           [Points[1][0],Points[1][1],x[2*i+1]],\
#                           Points[1] ] )
        Points = np.array( [    Points[0],\
                               [Points[0][0],Points[0][1],x[5*i]],\
                               [x[2*i+1],x[5*i+2],x[5*i+3]],\
                               [Points[1][0],Points[1][1],x[5*1+4]],\
                                Points[1]   ])
        
    
        ### The Points are stored in the array waveguides_matched that will 
        ### contain the control points of all the waveguides
        waveguides.append(Points)
    
    Points_ctl = np.array(waveguides)
    
    return Points_ctl


#############################################################################
### Opt





def cost(x,show=True):
    '''
    Cost is a function that weights the cost of proximity.
    
    x are the proximities for the different curves 
    alpha is the weighting factor, to be determined
    '''
    #return np.random.rand()
    # Get all the points for the input and variables given
    #print '...'
    P = guess_points(in_put,x)
    
    # For the guessed point calculate the parameters that will define the 
    # merit function
    n = len(P)

    for i in range(n):
        Pt = P[i]
    # Calculate the proximity, and take the first 10 smallest values
    prox = []
    prox_small = []

    for i in range(n):
        for j in range(i+1,n):
            l = proximity2(t,P[i],P[j])
            prox.append(l[1])
            prox_small.append(nsmallest(10,l[1]))
    prox = np.array(prox)
    prox_f = nsmallest(10,prox_small)
    prox_f = np.mean(prox_f)
    
    # caluclate the pathelength and take the standard deviation
    pathlength = []
    for i in range(n):
        pathlength.append(arc_length(P[i]))
    pathlength = np.array(pathlength)
    
    stand_dev = np.std(pathlength)
    
    # calculate the bend radius and take the 10 smallest values
    bend_radii = []
    bend_small = []
    for i in range(n):
        bend_radii.append(1.0/Curvature(P[i]))
        bend_small.append(nsmallest(10, 1.0/Curvature(P[i])))
    bend_radii_f = nsmallest(10,bend_small)
    bend_radii_f = np.mean(bend_radii_f)

    # calculate the different coefficient for weigthing the different
    # paramaters in the merit function 
    
    # first define a 'bad' and a 'good' value for each parameter
    x_bend = np.array([30000.,100000.])
    x_prox = np.array([300.,700.])
    x_stand = np.array([1.,0.1])

    # choose a 'high' and 'low' value for the merit function 
    #y = np.array([0.14,0.0025])  
    y = np.array([1.,0.1])

    
    def exp_coeff(y,x):
        a = ( np.log(1./y[1]) - np.log(1./y[0]) ) / ( x[1] - x[0] )
        b = np.log(1./y[0]) - a* x[0]
        return np.array([a,b])
    def quad_coeff(y,x):
        a = (y[1] - y[0]) / (x[1]**2 -x[0]**2)
        b = y[0] - a* x[0]**2
        return np.array([a,b])
    
    bend_coeff = exp_coeff(y,x_bend)
    prox_coeff = exp_coeff(y,x_prox)
    stand_coeff = quad_coeff(y,x_stand)

#bend_ = np.concatenate(np.random.rand(1,7000) * 50000)
#prox_ = np.concatenate(np.random.rand(1,7000) * 50)
#stand_ = np.concatenate(np.random.rand(1,7000))


    f = 1./np.exp(bend_coeff[0] * bend_radii_f + bend_coeff[1]) 
        #0.1*(stand_coeff[0] * (stand_dev)**2 + stand_coeff[1])
        #0.5/np.exp(prox_coeff[0] * prox_f + prox_coeff[1] )+ \
        
    print bend_radii_f
    print f
    print
#        1.*(stand_coeff[0] * stand_dev**2 + stand_coeff[1])
        
#    f=    1./np.exp(stand_coeff[0] * (1.-stand_dev) + stand_coeff[1])

    #print 'test 1 ',1./np.exp(stand_coeff[0] * (1.-0.9) - stand_coeff[1])
    #print 'test 2 ',1./np.exp(stand_coeff[0] * (1.-0.1) - stand_coeff[1])    
#    f = 1. / np.exp(4.605e-4 * bend_radii_f - 11.515)  + \
#        1./ np.exp(0.4605 * prox_f - 11.515 ) + \
#        1./ np.exp(2.3025 * (1./stand_dev) - 0.0025)

    return f

##############################################################################
### Run the optimisation with 'good' initial conditions

variables = []
points = []
merit_function = []

variables_0 = np.array( [    12985.64541354,\
                             2616.,   -372.725,  16120.4520603,\
                             18895.03267691,\
                
                             8803.15779172,\
                             2673.,   -312.125,  14552.46744791,\
                             20725.54338015,\
                             
                             9327.35060436,\
                             2574.5,   -327.275,  14569.94039492,\
                             20227.55998801,\
                
                             12362.89501789,\
                             2447.5,   -372.725,  16071.82612261,\
                             19444.84416603   ])
   
#bnds = [0., 30800.]

#bnds = (bnds, bnds, bnds, bnds, bnds, bnds, bnds, bnds)


#sol = scipy.optimize.minimize(cost,\
#                              variables_0,\
#                              method='Powell',bounds= bnds)
 
sol = scipy.optimize.minimize(cost,\
                              variables_0,\
                              method='Powell')

  
#points = guess_points(in_put,sol.x)


#
#    merit_function.append(sol.fun)
#    points.append(P)
#    variables.append(variables_0)
#    
#    np.save('initial_conditions',variables)
#    np.save('Bezier_points',points)
#    np.save('merit_function',merit_function)

##############################################################################
### Run the optimisation with random initial condtitions

#variables = []
#points = []
#merit_function = []
#
#for i in range(200):
#    variables_0 = np.random.rand(1,8) * 30000.
#    
#    
#    bnds = [0., 30000.]
#    #bnds_L = [25000., 35000.]
#
#    bnds = (bnds, bnds, bnds, bnds, bnds, bnds, bnds, bnds)
#
##def callbackF(x):
##    print cost(x,a,show=True)
##    print x
#
#    sol = scipy.optimize.minimize(cost,\
#                                  variables_0,\
#                                  args = (3.),\
#                                  method='Powell',bounds= bnds)
#    
#    P = guess_points(in_put,sol.x)
#

#
#    merit_function.append(sol.fun)
#    points.append(P)
#    variables.append(variables_0)
#    
#    np.save('initial_conditions',variables)
#    np.save('Bezier_points',points)
#    np.save('merit_function',merit_function)
#
#
###############################################################################
#### load the solution
#
#
#in_cdt = np.load('initial_conditions_0.npy')
#points = np.load('Bezier_points_0.npy')
#merit_fun = np.load('merit_function_0.npy')       
#
##### delete the solution which give a merit function greater than 100
#indices = [i for i,v in enumerate(merit_fun >= 100) if v]
# 
#merit_fun = np.delete(merit_fun,indices,0)
#points = np.delete(points,indices,0)
#
##ind = np.where(merit_fun == np.max(merit_fun))[0][0]
##
##merit_fun = np.delete(merit_fun,ind,0)
##points = np.delete(points,ind,0)
#
###############################################################################
#### plot merit function as function of the different opt parameters
#
#m = len(points)
#
#prox_ = []
#bend_ = []
#stand_ = []
#
#for k in range(m):
#    
#    P = points[k]
#    n = len(P)
#    
#    prox = []
#    for i in range(n):
#        for j in range(i+1,n):
#            l = proximity2(t,P[i],P[j])
#            prox.append(np.min(l[1]))
#    prox_ .append(np.min(prox))
#
#    pathlength = []
#    for i in range(n):
#        pathlength.append(arc_length(P[i]))
#    stand_.append(np.std(pathlength))
#    
#    bend_radii = []
#    for i in range(n):
#        bend_radii.append(np.min(1.0/Curvature(P[i])))
#    bend_.append(np.min(bend_radii))
#
#values= merit_fun
#fig = plt.figure(1)
#ax = fig.add_subplot(111,projection='3d')
#p = ax.scatter(prox_,np.array(bend_)/1000.,stand_, c = values)
##ax.set_zlim3d(0, 1000)
#ax.set_xlabel('prox')
#ax.set_ylabel('bend')
#ax.set_zlabel('stand')
#fig.colorbar(p, ax=ax)
#plt.show()   

############################################################################
#### Convert to n+1 = 5 Bezier control points
#    
#points = np.array([[[   105.        ,    -45.45      ,      0.        ],
#                    [   105.        ,    -45.45      ,  17314.19388472],
#                    [  5127.        ,   -700.        ,  14926.71023588],
#                    [  5127.        ,   -700.        ,  30800.        ]],
#
#                   [[   -35.        ,     75.75      ,      0.        ],
#                    [   -35.        ,     75.75      ,  11737.5437223 ],
#                    [  5381.        ,   -700.        ,  17367.39117353],
#                    [  5381.        ,   -700.        ,  30800.        ]],
#
#                   [[  -105.        ,     45.45      ,      0.        ],
#                    [  -105.        ,     45.45      ,  12436.46747248],
#                    [  5254.        ,   -700.        ,  16703.41331735],
#                    [  5254.        ,   -700.        ,  30800.        ]],
#                    
#                   [[  -105.        ,    -45.45      ,      0.        ],
#                    [  -105.        ,    -45.45      ,  16483.86002385],
#                    [  5000.        ,   -700.        ,  15659.79222137],
#                    [  5000.        ,   -700.        ,  30800.        ]]])
#
#
#points_5 = []    
#for i in range(len(points)):
#    points_5.append(pts_n_plus_1(points[i]))  
#points_5 = np.array(np.concatenate(points_5))
#print points_5  
#
#fig = plt.figure(1)
#ax = fig.add_subplot(111,projection='3d')
#for i in range(n):
#    ax.plot(Bezier(t,points[i][:,0]),\
#            Bezier(t,points[i][:,1]),\
#            Bezier(t,points[i][:,2]))
#    ax.plot(Bezier(t,points_5[i][:,0]),\
#            Bezier(t,points_5[i][:,1]),\
#            Bezier(t,points_5[i][:,2]))
##waveguides_matched = np.array(points)
##print 'Waveguides matched ',waveguides_matched
#ax.set_ylim3d(-2500, 2500)
##ax.set_xlabel('X')
##ax.set_ylabel('Y')
##ax.set_zlabel('Z')
##ax.set_axis_off()
#plt.show()

###########################################################################
### Enter the winning index, plot all you need to know about the winning 
### configuration 


#points = points_5

n = len(points)
            
pathlength = []
for i in range(n):
    pathlength.append(arc_length(points[i]))
    stand_dev = np.std(pathlength)
print 'stand dev',stand_dev
    #stand_dev_v.append(stand_dev)

stand_dev = np.std(pathlength)


#def match_guide(Points,Length):
#    '''
#    Points the input and output points for the waveguide
#    Length the length to match the pathlength to 
#    
#    return the ctl points for a pathlength match curved defined as a 
#    Bezier curve
#    '''
#    wave_guide_length = arc_length(Points)
#    print(wave_guide_length)
#    path_to_match = Length
#    print(path_to_match)
#    #L = 30793.68 # Length of the chip
#    L = 30000.
#    epsilon = 1.
#    low = 0.0
#    high = L
#    
#    
#    P = Points
#    print 'points ',P
#    while abs(wave_guide_length - path_to_match) > epsilon:
#        guess = low + abs(high-low)/2
#        P = np.array( [ Points[0],\
#                        [Points[1][0],Points[1][1],Points[1][2] + guess],\
#                        [Points[2][0],Points[2][1],Points[2][2] - guess],\
#                        Points[3] ] )
#        wave_guide_length = arc_length(P)    
#        if wave_guide_length - path_to_match < 0:
#            low = guess
#        elif wave_guide_length - path_to_match > 0:
#            high = guess
#        else:
#            break
#
#    return P

#points_matched = []
#for i in range(len(points)):
#    P = match_guide(points[i],np.max(pathlength))
#    points_matched.append(P)
#    
#   
#points = points_matched 
#
bend_radii = []
for i in range(n):
    bend_radii.append(1.0/Curvature(points[i]))
bend_radii = np.array(bend_radii)
min_bends = np.min(bend_radii)


fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
for i in range(n):
    ax.plot(Bezier(t,points[i][:,0]),\
            Bezier(t,points[i][:,1]),\
            Bezier(t,points[i][:,2]))
#waveguides_matched = np.array(points)
#print 'Waveguides matched ',waveguides_matched
ax.set_ylim3d(-2500, 2500)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
ax.set_axis_off()
plt.show()

constraint = 30000* np.ones(len(t))    
fig = plt.figure(2)
for i in range(n):
    plt.plot(bend_radii[i]) 
plt.plot(constraint)
plt.ylim([0, 150000])
plt.show()
print 'min bend',np.min(bend_radii)


constraint = 30 * np.ones(len(t)) 
fig = plt.figure(3)
for i in range(n):
    for j in range(i+1,n):
        l = proximity2(t,points[i],points[j])
        lab= 'waveguide ' + str(i+1) + ' and waveguide ' + str(j+1)
        plt.plot(l[0],l[1], label= lab)
        print 'prox', np.min(l[1])
plt.plot(l[0], constraint, label= 'please work!')
plt.legend()
plt.ylim([-100,1000])      
#plt.ylabel('Minimum distance (microns)')
plt.show()
#
#pathlength = []
#for i in range(n):
#    pathlength.append(arc_length(points[i]))
#pathlength = np.array(pathlength)
#
#pathlength = []
#for i in range(n):
#    pathlength.append(arc_length(points[i]))
#    stand_dev = np.std(pathlength)
#print 'stand dev',stand_dev
#    #stand_dev_v.append(stand_dev)
#
#stand_dev = np.std(pathlength)
    
    

        
t1 = datetime.datetime.now()
print 'Computation time ',t1-t0

