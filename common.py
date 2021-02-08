# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:44:18 2018

@author: threl300697
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
#from test_example import analytical_solution

a = 0
b = 1
#d = 0.025
gamma = -2
omega = 3*math.pi
Nmax = 75

#returns kernel function K(x,y)
def kernel(x,y,d):
    return d/((d**2 + (y-x)**2)**(1.5))

########################################################################################
#Chebyshev function
def chebyshev(a, b, N):
    # return Chebyshev's interpolation points on the interval [a,b]
    I = np.arange(1, N+1, 1)
    X = (b + a)/2 + (b - a)/2*np.cos((2*I - 1)*np.pi/(2*N))
    return X

##########################################################################################
#Make Gauss_Legendre quadrature nodes and weights
def gauss(a,b,N):
    t,v = np.polynomial.legendre.leggauss(N)
    #Flytter intervall fra [-1,1] til [a,b]
    xq = ((b-a)*t +(a+b))*0.5
    w =(b-a)*0.5*v
    return xq,w

#########################################################################################
#Lagrange polynomial
#returns a list
'''
def lagrange(N,x,Xb):
    Li=np.zeros(N)
    for i in range(N):
    # form the Lagrange polynomial i
        Li[i] = np.prod(x-Xb[np.arange(N)!=i])/np.prod(Xb[i]-Xb[np.arange(N)!=i])
    return Li 
'''
def lagrange1(x,xs,j):
    lagrange_prod = 1
    for i in range(len(xs)):
        if (i !=j):
            lagrange_prod *= (x - xs[i])/(xs[j]-xs[i])
    return lagrange_prod

###########################################################################################
#Midpoint nodes and weights
def xqList(a,b,n):
    h=(b-a)/n
    w = np.zeros(n)
    xq = np.zeros(n)
    for i in range(n):
        #xq.append(a+ i*h + h/2)
        xq[i]=a+i*h+h/2
        w[i] = h
    return xq,w
########################################################################################
#### Right hand side b=F

def fredholm_rhs(xc,d):
    #try:
    F = pickle.load( open( "F.pkl", "rb" ) )
   # except:
        #F = analytical_solution(a,b,omega,gamma,Nmax)   
    F_eval = F(xc,d)
    return F_eval

def fredholm_RHS(xc,d,F):
    #try:
   # except:
        #F = analytical_solution(a,b,omega,gamma,Nmax)   
    F_eval = F(xc,d)
    return F_eval
#######################################################################################
#####Left hand side A
'''
def fredholm_lhs(d,xc,xs,xq,w):
    Nc = xc.shape[0]
    Ns = xs.shape[0]
    Nq = xq.shape[0]
    A = np.zeros((Nc,Ns))
    ker = np.zeros(Nq)
    Li = np.zeros(Nq)
    for i in range(Nc):
        for j in range(Ns):
            for k in range(Nq):
                ker[k] = kernel(xc[i],xq[k],d)
                Li[k] = lagrange(Ns,xq[k],xs)[j] 
            A[i,j]= np.sum(ker*Li*w)
    return A
'''

def fredholm_LHS(d,xc,xs,xq,w):
    Nc = xc.shape[0]
    Ns = xs.shape[0]
    Nq = xq.shape[0]
    A = np.zeros((Nc,Ns))
    Lj = np.zeros(Nq)
    ker = np.zeros(Nq)
    for j in range(Ns):
        Lj = lagrange1(xq,xs,j)
        for i in range(Nc):
            #for k in range(Nq):
                #ker[k] = kernel(xc[i],xq[k],d)
            ker = kernel(xc[i],xq,d)
            A[i,j]= np.sum(ker*Lj*w)
    return A
