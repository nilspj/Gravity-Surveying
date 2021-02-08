# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:28:42 2018

@author: threl300697
"""

import numpy as np 
import matplotlib.pyplot as plt
import pickle
from test_example import analytical_solution

#different values of depth, d
d1= 0.025
d2= 0.25
d3= 2.5

#function \rho(x)
def p(x):
    if x > 1/3 and x < 2/3:
        return 1
    return 0

#antiderivative of K(x,y)
def antiderivative(x,y,d):
    return (y-x)/(d*(d**2 + (x-y)**2)**0.5)


#returns F(x)
def function(x,d): 
    return antiderivative(x,2/3,d) - antiderivative(x,1/3,d)
    

plt.subplots_adjust(hspace=0.4)
x2 = np.arange(0,1, 0.01)

#Plot with logarithmic y-axis for different d

plt.semilogy(x2, function(x2,d1),label=r'$d_{1}$')     #plot for depth d1
plt.semilogy(x2, function(x2,d2),label=r'$d_{2}$')     #plot for depth d2
plt.semilogy(x2, function(x2,d3),label=r'$d_{3}$')     #plot for depth d3

plt.ylabel(r"$F(x) \,[\mathrm{N}]$", fontsize=14)
plt.xlabel(r"$x \,[\mathrm{m}]$", fontsize=14)
plt.xlim(0,1)
plt.ylim(0,10**(2))
plt.legend(loc=1,fontsize=16)
plt.grid()
#plt.savefig('Question1.pdf')
plt.figure()