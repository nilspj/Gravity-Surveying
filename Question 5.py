# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
import common as co
import pickle
from test_example import analytical_solution

a = 0
b = 1
d1 = 0.025
d2 = 0.25
d3 = 2.5
Nc = np.array([])
Ns = np.array([])
gamma = -2
omega = 3*math.pi
Nmax = 75

def testp(x):
    return (math.sin(x*3*math.pi))*(math.exp(-2*x))

def Vectorp(xs):
    p = np.zeros(len(xs))
    for i in range(len(xs)):
        p[i] = testp(xs[i])
    return p


###########Test##########
'''
xq, w = co.xqList(a,b,1)
A = co.fredholm_lhs(xc,xs,xq,w)
print(np.linalg.solve(A,F))
''' 

#make our error vector analytical - numerical with gauss quadrature
def numError5(Nc,Ns,d):
    Ap = np.array([])
    Nclist = np.array([])
    F = pickle.load( open( "F.pkl", "rb" ) )
    for Nc in range(5,31):
        ###### i=1 gir "error singular matrix", begynn fra i=2#######
        xc = co.chebyshev(a,b,Nc)
        xs = xc
        Ns = Nc
        #F = (co.fredholm_rhs(xc,d))
        Feval = (co.fredholm_RHS(xc,d,F))
        p = Vectorp(xs)
        ##############################################
        
         ########## Gauss-Legendre noder t  , vekter v ###################
        #t, v = np.polynomial.legendre.leggauss(Nc**2)
        #Flytter intervall fra [-1,1] til [a,b]
        #xq = ((b-a)*t +(a+b))*0.5
        #w =(b-a)*0.5*v
        xq,w = co.gauss(a,b,Nc**2)
        
        '''
        xq, w = co.xqList(a,b,Nc**2)
        '''
        #A = co.fredholm_lhs(d,xc,xs,xq,w)
        A = co.fredholm_LHS(d,xc,xs,xq,w)
        ################################################
         ###Løser lignings systemet
        solP = np.linalg.solve(A,Feval)
        Error = p-solP
        Max = np.linalg.norm(Error,np.Inf)
        Ap = np.append(Ap, Max)
        Nclist = np.append(Nclist,Nc)
    return Ap,Nclist

#plot for different d
Ap1,Nclist1 = numError5(Nc,Ns,d1)
Ap2,Nclist2 = numError5(Nc,Ns,d2)
Ap3,Nclist3 = numError5(Nc,Ns,d3)
plt.semilogy(Nclist1,Ap1,label=r'$d_{1}$')
plt.semilogy(Nclist2,Ap2,label=r'$d_{2}$')
plt.semilogy(Nclist3,Ap3,label=r'$d_{3}$')
plt.ylabel(r"$Error \,[\mathrm{kg/m^3}]$", fontsize=13)
plt.xlabel(r"$Nc$",fontsize=14)
plt.xlim(5,30)
#plt.ylim(0,50)
plt.legend(loc=1,fontsize=12)
plt.grid()
#plt.savefig('Question5.pdf')
plt.figure()
