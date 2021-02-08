# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import common as co
import pickle
from test_example import analytical_solution

#Opening each file and unloading its contents
#produce our source and collocation points
#make our nodes and weights using gauss quadrature
#make our matrix using fredholm RHS
############################################
## File 1 ######
file_name1 = 'q8_1.npz'
f1         = open(file_name1,'rb')
npzfile1   = np.load(f1)

a1=npzfile1['a']
b1=npzfile1['b']
d1=npzfile1['d']
xc1=npzfile1['xc']
xs1 = co.chebyshev(a1,b1,len(xc1))
F1=npzfile1['F']
Nc1 = len(xc1)
Ns1 = len(xs1)
xq1,w1 = co.gauss(a1,b1,30)
A1 = co.fredholm_LHS(d1,xc1,xs1,xq1,w1)
############################################
## File 2 ######
file_name2 = 'q8_2.npz'
f2         = open(file_name2,'rb')
npzfile2   = np.load(f2)

a2=npzfile2['a']
b2=npzfile2['b']
d2=npzfile2['d']
xc2=npzfile2['xc']
xs2 = co.chebyshev(a2,b2,len(xc2))
F2=npzfile2['F']
Nc2 = len(xc2)
Ns2 = len(xs2)
xq2,w2 = co.gauss(a2,b2,30)
A2 = co.fredholm_LHS(d2,xc2,xs2,xq2,w2)
############################################
## File 3 ######
file_name3 = 'q8_3.npz'
f3         = open(file_name3,'rb')
npzfile3   = np.load(f3)

a3=npzfile3['a']
b3=npzfile3['b']
d3=npzfile3['d']
xc3=npzfile3['xc']
xs3 = co.chebyshev(a3,b3,len(xc3))
F3=npzfile3['F']
Nc3 = len(xc3)
Ns3 = len(xs3)
xq3,w3 = co.gauss(a3,b3,30)
A3 = co.fredholm_LHS(d3,xc3,xs3,xq3,w3)
##################################################

#returns a vector with a numerical solution \rho(x) using just one value of lambda
def smallError(Nc,d,l,A,F):
    At = np.transpose(A)
    RHS = np.matmul(At,F)
    LHS = np.matmul(At,A) + l*np.eye(Nc)
    solp = np.linalg.solve(LHS,RHS)
    return solp

#Plot our solutions \rho(x) for the different files
plt.plot(xs1,smallError(Nc1,d1,10**(-4),A1,F1),label=r'$p_{\lambda}1$')
plt.plot(xs2,smallError(Nc2,d2,10**(-5),A2,F2),label=r'$p_{\lambda}2$')
plt.plot(xs3,smallError(Nc3,d3,10**(-4),A3,F3),label=r'$p_{\lambda}3$')
plt.ylabel('p')
plt.xlabel('$x$')
plt.legend(loc=1)
plt.grid()
plt.figure()