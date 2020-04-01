#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:23:00 2017

@author: tiphainelagadec
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:04:15 2017

@author: tiphainelagadec
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:46:28 2017

@author: tiphainelagadec
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import datetime
import scipy.optimize

plt.close('all')
t0 = datetime.datetime.now()

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 15}

#############################################################################
# Parameters (all lengths in microns)

# x is the horizontal axis, y the vertical and z the depth

L = 30793.68 # Length of the remaping chip

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
    x_offset = 5000.0
    step = 127
    x1 = center_chip(inputPoints) + x_offset
    
    for i in range(len(inputPoints)):
        out_put.append([x1,0,L])
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
    
    connect = np.array([[Input[0],Output[3]],\
                        [Input[1],Output[2]],\
                        [Input[2],Output[1]],\
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
    #L = 30800.
    #L = 31000.
    L = x[8]
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
        Points = np.array( [ Points[0],\
                        [Points[0][0],Points[0][1],x[2*i]],\
                        [Points[1][0],Points[1][1],x[2*i+1]],\
                        Points[1] ] )
        ### The Points are stored in the array waveguides_matched that will 
        ### contain the control points of all the waveguides
        waveguides.append(Points)
    
    Points_ctl = np.array(waveguides)
    
    return Points_ctl


#############################################################################
### Opt





def cost(x,a,b,c,show=True):
    '''
    Cost is a function that weights the cost of proximity.
    
    x are the proximities for the different curves 
    alpha is the weighting factor, to be determined
    '''
    #return np.random.rand()
    # Get all the points for the input and variables given
    P = guess_points(in_put,x)
    
    
    # n is the number of waveguides
    n = len(P)
    
#    
#    fig1 = plt.figure(1)
#    ax = fig1.add_subplot(111,projection='3d')
    for i in range(n):
        Pt = P[i]
#        #print(i)
#        #print(ctl_pts)
#        ax.plot(Bezier(t,Pt[:,0]),\
#                Bezier(t,Pt[:,1]),\
#                Bezier(t,Pt[:,2]))
#    ax.set_zlim3d(-15000, 15000)
#    ax.set_ylim3d(0,30000)
#    ax.set_xlim3d(0,30000)
#    ax.set_xlabel('X')
#    ax.set_ylabel('Y')
#    ax.set_zlabel('Z')
#    plt.rc('font', **font)
#    plt.show()
        
    
    
    # Calulate the minimum proximities
    min_prox = []
    
#    fig2 = plt.figure(2)
#    print
#    print 'Minimum proximity '
    for i in range(n):
        for j in range(i+1,n):
            l = proximity2(t,P[i],P[j])
#            print np.min(l[1])
            #print 'Minimum proximity ',l
#            lab= 'waveguide ' + str(i+1) + ' and waveguide ' + str(j+1)
#            plt.plot(l[0],l[1], label= lab)
#            plt.legend()
            min_prox.append(np.min(l[1]))
    min_prox = np.array(min_prox)
#    plt.ylim([-100,400])      
#    plt.ylabel('Minimum distance (microns)')
#    plt.show()

    
    
    # Calculate the pathlength matching
    pathlength = []
    for i in range(n):
        pathlength.append(arc_length(P[i]))
    pathlength = np.array(pathlength)
#    print 'Pathlength ',pathlength
    
    
    
    #Calculate the Bend radius
#    fig3 = plt.figure(3)
    bend_radii = []
#    print
#    print 'Minimum bend radius '
    for i in range(n):
        bend_radii.append(1.0/Curvature(P[i]))
#        plt.plot(1.0/Curvature(P[i])*1e-3)
#        print min(1.0/Curvature(P[i])*1e-3)
    bend_radii = np.array(bend_radii)
#    plt.ylabel('radius of curvature along the waveguide (mm)')
#    #plt.ylim([0,200]) 
#    plt.show()

    ### Pathlength matching opt
    ### stand deviation is in microns
    stand_dev = np.std(pathlength)
        
#    f = 0.04*np.exp(0.01*stand_dev) +\
#        1/ np.exp(0.05*np.min(bend_radii) ) + \
#        1/ np.exp(0.2* np.min(min_prox))


    f = 1. * stand_dev/a - \
        1. * np.min(bend_radii)/b - \
        1. * np.min(min_prox)/c
#
#    f = 1.0 * stand_dev/a   
#    f = np.exp(a  * stand_dev) - \
#        np.exp(1.e-4 * np.min(bend_radii)) - \
#        np.exp(1.e-4  * np.min(min_prox))
        
#    print f
#    print 'test '
#    print stand_dev
#    print np.min(bend_radii)
#    print np.min(min_prox)

    return f


variables_0 = np.array([ 10266.67, 20533.33,\
                         10266.67, 20533.33,\
                         10266.67, 20533.33,\
                         10266.67, 20533.33,
                         30000.])
#    
variables_3210 = np.array([  9502.83187063,  20879.30494117,\
                             9541.86142078,  21087.21190549,\
                             10203.72303157, 20602.91446772,\
                             12549.01148658,  19698.77267124,\
                             30074.72234913])
#    
variables_004 = np.array([ 9578.26, 20634.95,  9814.09,  21600.87,\
                           10482.19,  20469.50,  12904.48,  20146.11,\
                           30190.31])

variables_new = np.array([  9016.30536085,  20767.37969346,\
                            8771.00732985,  20865.51060964,\
                            9996.53517647,  20716.78378346,\
                            11690.04487221,  19008.08279197,\
                            29950.9468878 ])
#variables_0 = np.array([ 15400., 15400.,\
#                         15400., 15400.,\
#                         30800. ])
    

#variables_0 = np.array([ 14781.44379587,  17344.66480295,\
#                         11963.26481406,  18832.01446681,\
#                         31042.95945015])

#variables_new = np.array([ 13504.82612251,  18354.63690933,  \
#14796.66952719,\
#  17905.02598295,
#        11109.05905418,  18324.89785126,  10978.46971902,  19487.72325751,
#        30505.33894529])
# 
#a = 1.5   
#l = cost(variables_new,a)
#print l

bnds = [0., 30000.]
bnds_L = [25000., 35000.]
#
##bnds = (bnds, bnds, bnds, bnds,bnds, bnds, bnds_L )
bnds = (bnds, bnds, bnds, bnds, bnds, bnds, bnds, bnds, bnds_L)

def callbackF(x):
    print cost(x,a,show=True)
    print x
#    
#test_curv = np.array( [[[ 2.60e+01, -4.50e+01,  0.00e+00],\
#                        [ 2.60e+01, -4.50e+01,  2.08e+04],\
#                        [ 6.50e+03,  0.00e+00,  9.95e+03],\
#                        [ 6.50e+03,  0.00e+00,  3.08e+04]],\
#                       [[ 0.00e+00,  9.00e+01,  0.00e+00],\
#                        [ 0.00e+00,  9.00e+01,  2.71e+04],\
#                        [ 5.00e+03,  0.00e+00,  3.72e+03],\
#                        [ 5.00e+03,  0.00e+00,  3.08e+04]],\
#                       [[ 5.20e+01,  6.00e+01,  0.00e+00],\
#                        [ 5.20e+01,  6.00e+01,  1.96e+04],\
#                        [ 6.75e+03,  0.00e+00,  1.12e+04],\
#                        [ 6.75e+03 , 0.00e+00,  3.08e+04]]])
#
#d1 = proximity(t,test_curv[0],test_curv[1])
#d2 = proximity2(t,test_curv[0],test_curv[1])
#
#plt.plot(d1[0],d1[1])
#plt.plot(d2[0],d2[1],'o')
#plt.show()


    #pts_search = []

#a = np.arange(0.1,10,1e-2)
#for i in range(len(a)):
#  
pts = []
min_bends_v = []
min_prox_v = []
stand_dev_v = []

a = np.arange(0.,1.5,0.015)
#a = np.array([0.67499999999999993])
#a = np.array([1.425])

b = np.arange(0.,1000.,10.)
#b = np.array([1.])

c = np.arange(0.,1000,10.)
#c = np.array([1.])

function = []
for i in range(len(a)):
    for j in range(len(b)):
        for k in range(len(c)):
            sol = scipy.optimize.minimize(cost,\
                                          variables_0,args=(a[i],b[j],c[k]),\
                                          method='TNC',bounds= bnds)
            
#    sol = scipy.optimize.minimize(cost,variables_0,args=(a,), method='TNC',\
#                                  bounds= bnds,\
#                                  options={'disp':True, 'maxiter':200},\
#                                  callback=callbackF)
            
        
            #print 'solution ',sol
            P = guess_points(in_put,sol.x)
            pts.append(P)
            # n is the number of waveguides
            n = len(P)
            
            # Calulate the minimum proximities
            min_prox = []
            for i in range(n):
                for j in range(i+1,n):
                    l = proximity2(t,P[i],P[j])
                    min_prox.append(np.min(l[1]))
            min_prox = np.array(min_prox)
            min_prox_v.append(np.min(min_prox))
            print 'min proximity ',min_prox_v
            
            # Calculate the pathlength matching
            pathlength = []
            for i in range(n):
                pathlength.append(arc_length(P[i]))
            pathlength = np.std(pathlength)
            stand_dev_v.append(pathlength)
            print 'stand deviation ',stand_dev_v
            
            #Calculate the Bend radius
            bend_radii = []
            for i in range(n):
                bend_radii.append(1.0/Curvature(P[i]))
            bend_radii = np.array(bend_radii)
            min_bends_v.append(np.min(bend_radii))
            print 'min bend radius ',min_bends_v
        
            function.append(sol.fun)
         
    
print
print 'out of the second loop'
min_prox_v = np.array(min_prox_v)
min_bends_v = np.array(min_bends_v)
stand_dev_v = np.array(stand_dev_v)
function = np.array(function)

np.save('proximities',min_prox_v)
np.save('bends',min_bends_v)
np.save('deviations',stand_dev_v)
np.save('meritfunction',function)

print 
print 'out of the third loop, finish !'
        
        
#l = cost(variables_new,a)

#print sol.x
#test = cost(sol.x,a)
#print test

    
#x = np.arange(0,100,0.001)
#x = np.arange(0.,100.,0.001)
##
#fig4 = plt.figure(4)
##plt.plot(x,1/np.exp(0.3*x))
#plt.plot(x,np.exp(x))
#plt.xlim([0, 300])
#plt.show()

#x = np.array([  1.03029898e+05,   2.71839194e+00,   6.74523162e-01])
#
#a = 0.001
#b = 50
#c = 50
#
##x = np.array([ 330.57271959,   31.83782326,   44.9538968 ])
##
##
#def f(x):
#    f = 1/ np.exp(x[0]*a)  + \
#        1/ np.exp(x[1]*b ) + \
#        1/ np.exp(x[2]*c)
#    return f
## 
#print f(x)
##sol = scipy.optimize.minimize(f,x, method='TNC')
#

t1 = datetime.datetime.now()
print 'Computation time ',t1-t0

