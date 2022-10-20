import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import train.data as data
    
"""
Utilities for plotting 3D plots and visualizing images (e.g., rectangles and digits)
"""

def plot3D(evidence,marginals,x,y,z):
    assert data.evd_var_count(evidence) == 2

    xlabel = '$\lambda_%s$' % x
    ylabel = '$\lambda_%s$' % y
    zlabel = '$P(%s)$'      % z
    
    [E1,E2], PR = data.data2fn(evidence,marginals)
    
    __plot(E1,E2,PR,xlabel,ylabel,zlabel)
    

# image is specified using lambdas
# reshape works since lambdas are ordered row, then column
def image_lambdas(lambdas,label,size):
    assert type(lambdas) is list
    lambdas = np.array(lambdas)
    assert lambdas.shape == (size*size,2)
    lambdas = lambdas[:,0]
    image_  = lambdas.reshape([size,size])
    image(image_,label)
    
def image(image,label):
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title(label)
    plt.show()
    
# X,Y,Z are 1D arrays
def __plot(X,Y,Z,xlabel,ylabel,zlabel): 
    fig = plt.figure()
    ax  = fig.gca(projection='3d')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=0) #plot the surface
    #ax.view_init(60, 60) #rotate it
    plt.show() #display it
    
