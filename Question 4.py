# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import common as co
import pickle
from test_example import analytical_solution

a = 0
b = 1
d = 0.025
Nc = 40
Ns = 40
gamma = -2
omega = 3*math.pi
Nmax = 75

def testp(x):
    return (math.sin(x*3*math.pi))*(math.exp(-2*x))

def Vectorp(xs):
    p = np.zeros(len(xs))
    for i in range(Ns):
        p[i] = testp(xs[i])
    return p

xc = co.chebyshev(a,b,Nc)
xs = co.chebyshev(a,b,Ns)
F = (co.fredholm_rhs(co.chebyshev(a,b,Nc),d))
p = Vectorp(xs)

'''
t, v = np.polynomial.legendre.leggauss(32)
#Flytter intervall fra [-1,1] til [a,b]
xq = ((b-a)*t +(a+b))*0.5
w =(b-a)*0.5*v

A = co.fredholm_lhs(d,xc,xs,xq,w)
Error = F - np.matmul(A,p)
#print(Error)
#print(np.linalg.norm(Error,np.Inf))
'''


Nq1 = np.array([])
Ap1 = np.array([]) 

for i in range(5,250,25):
    ########## Gauss-Legendre noder t  , vekter v ###################
    print(i)
    t, v = np.polynomial.legendre.leggauss(i)
    #Flytter intervall fra [-1,1] til [a,b]
    xq = ((b-a)*t +(a+b))*0.5
    w =(b-a)*0.5*v
    #############################################
    #A = co.fredholm_lhs(d,xc,xs,xq,w)
    A = co.fredholm_LHS(d,xc,xs,xq,w)
    Nq1 = np.append(Nq1, i)
    Error = F - np.matmul(A,p)
    Max = np.linalg.norm(Error,np.Inf)
    Ap1 = np.append(Ap1, Max)
    

plt.semilogy(Nq1,Ap1)
plt.ylabel('Error')
plt.xlabel('Nq')
#plt.xlim(1,16)
#plt.ylim(0,50)
plt.legend(loc=1)
plt.grid()
plt.figure()

