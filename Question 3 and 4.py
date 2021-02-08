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

##Testfunksjon \rho(x) analytical
def testp(x):
    return (math.sin(x*3*math.pi))*(math.exp(-2*x))

#gir analytical p(xs) som liste
def Vectorp(xs):
    p = np.zeros(len(xs))
    for i in range(Ns):
        p[i] = testp(xs[i])
    return p

#make collocation and source points
xc = co.chebyshev(a,b,Nc)
xs = co.chebyshev(a,b,Ns)

'''
Hva vi trenger:
    array av max|F - Ap| for hver komponent c, med hensyn på Nq
    en vector av Ap for hver Nq
    en funksjon som regner ut F-Ap for hver Nq
    en funksjon som regner ut F-Ap for en Nq først
    og finn max verdien
'''

#make vectors b and \rho
F = (co.fredholm_rhs(xc,d))
p = Vectorp(xs)

Nq1 = np.array([])
Ap1 = np.array([]) 

##print test av matrise og største error
'''
xq,w = co.xqList(a,b,32)
#A = co.fredholm_lhs(d,xc,xs,xq,w)
A = co.fredholm_LHS(d,xc,xs,xq,w)
Error = F - np.matmul(A,p)
print(Error)
print(np.linalg.norm(Error,np.Inf))
'''
#make the error vector F - Ap with midpoint
for i in 5*2**np.arange(0,8):
    xq, w = co.xqList(a,b,i)
    #A = co.fredholm_lhs(d,xc,xs,xq,w)
    A = co.fredholm_LHS(d,xc,xs,xq,w)
    Nq1 = np.append(Nq1, i)
    Error = F - np.matmul(A,p)
    Max = np.linalg.norm(Error,np.Inf)
    Ap1 = np.append(Ap1, Max)
   
Nq2 = np.array([])
Ap2 = np.array([]) 

#make the error vector F - Ap with Gauss
for i in range(5,275,25):
    ########## Gauss-Legendre noder t  , vekter v ###################
    #t, v = np.polynomial.legendre.leggauss(i)
    #Flytter intervall fra [-1,1] til [a,b]
    #xq = ((b-a)*t +(a+b))*0.5
    #w =(b-a)*0.5*v
    xq,w = co.gauss(a,b,i)
    #############################################
    #A = co.fredholm_lhs(d,xc,xs,xq,w)
    A = co.fredholm_LHS(d,xc,xs,xq,w)
    Nq2 = np.append(Nq2, i)
    Error = F - np.matmul(A,p)
    Max = np.linalg.norm(Error,np.Inf)
    Ap2 = np.append(Ap2, Max)

plt.semilogy(Nq1,Ap1,label=r'Midpoint')
plt.semilogy(Nq2,Ap2,label=r'Legendre–Gauss')
plt.ylabel(r"$Error \,[\mathrm{N}]$", fontsize=14)
plt.xlabel(r"$Nq$",fontsize=14)
plt.xlim(5,250)
#plt.ylim(0,50)
plt.legend(loc=1,fontsize=14)
plt.grid()
#plt.savefig('Question3&4.pdf')
plt.figure()
