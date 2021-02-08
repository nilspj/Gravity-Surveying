# -*- coding: utf-8 -*-
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

#make our perturbed and non-perturbed vectors b
F1 = (co.fredholm_RHS(xc,d1,F))
F2 = (co.fredholm_RHS(xc,d2,F))
F3 = (co.fredholm_RHS(xc,d3,F))

B1 = F1*(1+epsilonVector(len(F1)))
B2 = F2*(1+epsilonVector(len(F2)))
B3 = F3*(1+epsilonVector(len(F3)))

p = Vectorp(xs)

Nq = np.array([])

###########noder og vekter with gauss quadrature
#xq, w = co.xqList(a,b,32)

#t, v = np.polynomial.legendre.leggauss(16)
#Flytter intervall fra [-1,1] til [a,b]
#xq = ((b-a)*t +(a+b))*0.5
#w =(b-a)*0.5*v
xq,w = co.gauss(a,b,30)


################ lager matriser A med forskjellig d
#A1 = co.fredholm_lhs(d1,xc,xs,xq,w)
#A2 = co.fredholm_lhs(d2,xc,xs,xq,w)
#A3 = co.fredholm_lhs(d3,xc,xs,xq,w)

A1 = co.fredholm_LHS(d1,xc,xs,xq,w)
A2 = co.fredholm_LHS(d2,xc,xs,xq,w)
A3 = co.fredholm_LHS(d3,xc,xs,xq,w)
###Løser lignings systemet for non-perturbed b
sP1 = np.linalg.solve(A1,F1)
sP2 = np.linalg.solve(A2,F2)
sP3 = np.linalg.solve(A3,F3)
####### løsning for perturbed b
sPx1 = np.linalg.solve(A1,B1)
sPx2 = np.linalg.solve(A2,B2)
sPx3 = np.linalg.solve(A3,B3)

########## b og \tilde{b} plot for different d
#plt.plot(xs,F1,'r',linewidth=2,label=r'Analytical')
#plt.plot(xs,B1,'b--',label=r'Perturbed')

#plt.plot(xs,F2,'r',linewidth=2, label=r'Analytical')
#plt.plot(xs,B2,'b--',label=r'Perturbed')

plt.plot(xs,F3*1000,'r',linewidth=2, label=r'Analytical')
plt.plot(xs,B3*1000,'b--',label=r'Perturbed')

############ \rho(x) for perturbed and non-perturbed b
#plt.plot(xs,sP1,'b--',linewidth=2,label=r'Non-perturbed')
#plt.plot(xs,sPx1,'r',linewidth=1,label=r'Perturbed')

#plt.plot(xs,sP2,'b--',linewidth=2,label=r'Non-perturbed')
#plt.plot(xs,sPx2,'r',linewidth=1,label=r'Perturbed')

#plt.plot(xs,sP3,'b--',linewidth=2,label=r'Non-perturbed')
#plt.plot(xs,sPx3,'r',linewidth=1,label=r'Perturbed')

#Our analytical \rho
#plt.plot(xs,p,'g',label=r'Analytical')

plt.ylabel(r"$F(x) \,[\mathrm{mN}]$", fontsize=14)
#plt.ylabel(r"$"+chr(961)+"(x) \,[\mathrm{kg/m^3}]$", fontsize=14)
plt.xlabel(r"$x \,[\mathrm{m}]$", fontsize=14)
plt.xlim(0,1)
#plt.ylim(-2*10**7,2*10**7)
plt.legend(loc=1,fontsize=14)
plt.grid()
#plt.savefig('Question6_3.pdf')
plt.figure()
