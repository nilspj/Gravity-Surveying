# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 20:19:14 2018

@author: threl300697
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import common as co
import pickle
from test_example import analytical_solution

a = 0
b = 1
d1= 0.025
d2 = 0.25
d3 = 2.5
Nc = 30
Ns = 30
gamma = -2
omega = 3*math.pi
Nmax = 75

def epsilonVector(n):
    epsilon = np.zeros(n)
    for i in range(n):
        epsilon[i] = np.random.uniform(-0.001,0.001)
    return epsilon 

def testp(x):
    return (math.sin(x*3*math.pi))*(math.exp(-2*x))

def Vectorp(xs):
    p = np.zeros(len(xs))
    for i in range(Ns):
        p[i] = testp(xs[i])
    return p

#make collocation and source points
xc = co.chebyshev(a,b,Nc)
xs = xc
F = pickle.load( open( "F.pkl", "rb" ) )

#make perturbed and non-perturbed b
F2 = (co.fredholm_RHS(xc,d2,F))
F3 = (co.fredholm_RHS(xc,d3,F))

B2 = F2*(1+epsilonVector(len(F2)))
B3 = F3*(1+epsilonVector(len(F3)))

#our analytical \rho
p = Vectorp(xs)

Nq = np.array([])

###########noder og vekter with gauss quadrature
#xq, w = co.xqList(a,b,32)

#t, v = np.polynomial.legendre.leggauss(16)
#Flytter intervall fra [-1,1] til [a,b]
#xq = ((b-a)*t +(a+b))*0.5
#w =(b-a)*0.5*v
xq,w = co.gauss(a,b,30)

################ make matrix A med forskjellig d
#A1 = co.fredholm_lhs(d1,xc,xs,xq,w)
#A2 = co.fredholm_lhs(d2,xc,xs,xq,w)
#A3 = co.fredholm_lhs(d3,xc,xs,xq,w)

A2 = co.fredholm_LHS(d2,xc,xs,xq,w)
A3 = co.fredholm_LHS(d3,xc,xs,xq,w)
###Løser lignings systemet for non-perturbed b
sP2 = np.linalg.solve(A2,F2)
sP3 = np.linalg.solve(A3,F3)
####### løsning for perturbed b
sPx2 = np.linalg.solve(A2,B2)
sPx3 = np.linalg.solve(A3,B3)

#transpose matrixes
A2t = np.transpose(A2)
RHS = np.matmul(A2t,B2)

A3t = np.transpose(A3)
RHS1 = np.matmul(A3t,B3)

#returns error between analytical and numerical for many lambda
def rhoError7(Nc,Ns,d):
    Ap = np.array([])
    Lambda = np.geomspace(10**(-14),10**(1),num=30)
    #print(Lambda)
    for i in range(len(Lambda)):
        LHS = np.matmul(A2t,A2) + Lambda[i]*np.eye(Nc)
        solp = np.linalg.solve(LHS,RHS)
        Error = p-solp
        Max = np.linalg.norm(Error,np.Inf)
        Ap = np.append(Ap, Max)
    return Ap,Lambda

#returns error between analytical and numerical for one lambda
def smallError(Nc,d,l):
    LHS = np.matmul(A2t,A2) + l*np.eye(Nc)
    solp = np.linalg.solve(LHS,RHS)
    return solp

#returns error between analytical and numerical for many lambda
def rhoError71(Nc,Ns,d):
    Ap = np.array([])
    Lambda = np.geomspace(10**(-14),10**(1),num=30)
    #print(Lambda)
    for i in range(len(Lambda)):
        LHS = np.matmul(A3t,A3) + Lambda[i]*np.eye(Nc)
        solp = np.linalg.solve(LHS,RHS1)
        Error = p-solp
        Max = np.linalg.norm(Error,np.Inf)
        Ap = np.append(Ap, Max)
    return Ap,Lambda

#returns error between analytical and numerical for one lambda
def smallError1(Nc,d,l):
    LHS = np.matmul(A3t,A3) + l*np.eye(Nc)
    solp = np.linalg.solve(LHS,RHS1)
    return solp

#Plotting our errors with respect to lambda and our \rho with the smallest error
plt.plot(xs,p,'r',label=r'Analytical')
plt.plot(xs,smallError(Nc,d2,10**(-4)),'b--',label=r'Regularized')
#plt.plot(xs,smallError1(Nc,d3,10**(-9)),'b',label=r'Regularized')

#plt.plot(xs,sP2,linewidth=4,label=r'$p_{2}$')
#plt.plot(xs,sPx2,linewidth=2,label=r'$\~{p}_{2}$')

#rhoError,Lambda = rhoError7(Nc,Ns,d2)
#plt.loglog(Lambda,rhoError,label=r'Maximum difference')
#rhoError1,Lambda1 = rhoError71(Nc,Ns,d3)
#plt.loglog(Lambda1,rhoError1,label=r'Maximum difference')

#plt.ylabel(r"$Error \,[\mathrm{kg/m^3}]$", fontsize=13)
plt.ylabel(r"$"+chr(961)+"(x) \,[\mathrm{kg/m^3}]$", fontsize=13)
#plt.xlabel('$\lambda$',fontsize=14)
plt.xlabel(r"$x \,[\mathrm{m}]$", fontsize=14)
#plt.xlim(10**(-14),10**(1))
plt.xlim(0,1)
plt.legend(loc=1,fontsize=14)
plt.grid()
#plt.savefig('Question7_3.pdf')
plt.figure()