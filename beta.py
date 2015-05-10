#!/bin/python2

import math
import matplotlib.pyplot as plot
from numpy import *

def beta(x,y):
    return math.gamma(x)*math.gamma(y)/math.gamma(x+y)

def betad(x,a,b):
    return (1.0/beta(a,b))*(x**(a-1))*((1-x)**(b-1))

def plotbetad(a,b):
    x = arange(0,1,0.01)
    y = betad(x,a,b)
    plot.plot(x,y)

def plot_with_sigma(sigma,c=1):
    a = (1/(1.0-sigma))**c
    b = (1.0-sigma)**(c-1)
    x = arange(0,1,0.01)
    y = betad(x,a,b)
    plot.plot(x,y)
    xy = (0.8,betad(0.8,a,b))
    plot.annotate("Sigma=%s" % (str(sigma)),xy=xy)


