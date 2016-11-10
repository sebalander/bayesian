# -*- coding: utf-8 -*-
'''
script para probar kalman sencillo en 1D y ver la convergencia y qué pasa
cuando el modelo es incorrecto

@author
'''
# %%
import numpy as np
import matplotlib.pyplot as pt
import pykalman as pk

# %%
# defino sistema sin excitación
A = 1
H = 1
Q = 0.1
R = 1
n = 50 # cantidad de pasos
ite = np.arange(n)

# condiciones iniciales y lista de estados para todo tiempo
x = np.zeros(n)
P = np.ones(n)+1
xp = np.zeros(n)
Pp = np.zeros(n)

# "realidad" y mediciones
v = 1 # velocidad
X = ite * v
ruido = np.random.rand(n) - 0.5
z = X + ruido

# %% bucle
A2 = A**2
H2 = H**2

for i in ite[1:]:
    # etapa de prediccion
    xp[i] = A * x[i-1]
    Pp[i] = A2 * P[i-1] + Q

    # auxilaires
    k = Pp[i] * H / (H2 * Pp[i] + R)
    nu = z[i] - H * xp[i]

    # etapa de correccion
    x[i] = xp[i] + k * nu
    P[i] = Pp[i] * (1 - k * H)

sig = np.sqrt(P)
e = x - X

# %% plot estimacion
pt.figure()
pt.title('Estimacion de estado')
pt.plot(X,'-k',label='realidad')
pt.plot(z,'-r',label='mediciones')
pt.errorbar(ite,x,yerr=sig,label='estimacion')
pt.legend(loc='best')

# %% plot errores
Pinf = (-Q + np.sqrt(Q**2 + 4 + R * Q)) / 2
sInf = np.sqrt(Pinf)

eInf = - v * (Pinf + Q + R) / (Pinf + Q)

pt.figure()
pt.title('desv estandar y erroes')

pt.plot([0,n],[sInf,sInf],'--b',label=r'$\sqrt{P_{inf}}$')
pt.plot([0,n],[-sInf,-sInf],'--b')
pt.plot(sig,'-b',linewidth=2,label=r'$\sqrt{P}$')
pt.plot(-sig,'-b',linewidth=2)
pt.fill_between(ite,-sig,sig,facecolor='b',alpha=0.4)

pt.plot([0,n],[eInf,eInf],'--k',label=r'$e_{inf}$')
pt.plot(e,'-k',linewidth=2,label=r'$e$')
pt.legend(loc=1)
