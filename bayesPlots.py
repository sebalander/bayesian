# %%

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

binom = sc.stats.binom

# %% https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html
fig, ax = plt.subplots(1, 1)

#Calculate a few first moments:


n, p = 2**8 - 1, 0.7


mean, var, skew, kurt = binom.stats(n, p, moments='mvsk')

#Display the probability mass function (pmf):

x = np.arange(binom.ppf(0.0, n, p), binom.ppf(1, n, p))

ax.plot(x, binom.pmf(x, n, p), 'bo', ms=8, label='binom pmf')

ax.vlines(x, 0, binom.pmf(x, n, p), colors='b', lw=5, alpha=0.5)



# %%
cen = 0.65
de = 0.1

a = cen*(cen*(1-cen)/de**2 - 1)
b = a * (1-cen)

mu = a/(a+b)
std = np.sqrt(a*b/(a+b)**2/(a + b + 1))


cen, de
a,b
mu, std



# %%
# intensity values of some pixel, between 0 and 1, as floats
N = 100

Y = sc.stats.uniform.rvs(size=N)*de*2 - de + cen
x = np.linspace(0, 1, 100)

# %%
beta = sc.stats.beta



a, b = 40 , 20
plt.plot(x, beta.pdf(x, a, b), 'r-', lw=5, alpha=0.6, label='beta pdf')
plt.hist(Y, normed=True)



# %%
gamma = sc.special.gamma

def P(Al,Be,Y):

    y1 = np.prod(Y)
    y2 = np.prod(1-Y)

    N = Y.shape[0]

    G = (gamma(Al+Be) / gamma(Al) / gamma(Be))**N

    return  G * y1**(Al - 1) * y2**(Be - 1)


# %%
# fit a beta via bayes
al = np.linspace(0.01,500,200)
be = np.linspace(0.01,500,200)

Al, Be = np.meshgrid(al,be)

p = P(Al,Be,Y)

# %%
from mpl_toolkits.mplot3d import axes3d

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_surface(Al, Be, p)
ax.set_xlabel('Alfa')
ax.set_ylabel('Beta')
plt.show()


# %%

years = 2016 - np.array([32, 26, 24, 23, 21, 18])
on = np.ones(years.shape)

plt.plot(years, on, 'bo', ms=8, label='binom pmf')
plt.vlines(years, 0, on, colors='b', lw=5, alpha=0.5)
plt.xlim([1980, 2016])
plt.ylim([0, 1.5])
plt.xlabel('Anio')
plt.show()
