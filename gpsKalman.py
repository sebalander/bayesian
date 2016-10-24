
# %%
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

## %% example from pykalman page http://pykalman.github.io/
#kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]],
#                  observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
#
#measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
#kf = kf.em(measurements, n_iter=5)
#(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
#(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)


# %% LOAD GPS DATA
dataFile = "/home/sebalander/Code/bayesian/data/20161006224131.txt"

data = np.loadtxt(dataFile,delimiter=",",skiprows=50,usecols=(1,2)).T
# pongo los datos medidos respecto a la media para que quede como referencia
# multiplico todo por 1e6 para que quede en nros mas manejables
data = (data.T - np.mean(data,axis=1)).T * 1e6

# tomo solo las coordenadas x como medicion
# se va a trabajar solo sobre esta componente
measurementsX = data[0]

plt.figure()
plt.plot(data[0],data[1])
plt.figure()
plt.plot(measurementsX)


# %% KALMAN PARAMETERS

# transition matrix. Continous model
# son las derivadas temporales de los estados
A = np.zeros((2,2))
# measurement matrix
H = np.ones(2)
# the vector state must be [x, rx], the position and the gps error
# transition covariance
q = np.cov(measurementsX) # covariance of all data
f = 1e-2 # weigth associated to actual position (very small)
g = np.sqrt(1-f**2)
Q = q*np.array([[f,0],[0,g]])
# observation covariance: zero because we measure the state of noise as it is?
R = 0
# initial state mean
mu_0 = [-200, -200]
# initial state covariance
Sigma_0 = np.eye(2,2)

# que cosas optimizar por EM
# ['transition_matrices',     'observation_matrices', 'transition_offsets',
# 'observation_offsets',     'transition_covariance', 'observation_covariance',
# 'initial_state_mean',     'initial_state_covariance'] or 'all'
em_vars = ['observation_covariance', 'transition_covariance'] #"all" # ['transition_covariance', 'observation_covariance',
           #'initial_state_mean',     'initial_state_covariance']

# %% defino el filtro
kf = KalmanFilter(transition_matrices=A,
                  observation_matrices=H,
                  transition_covariance=Q,
                  observation_covariance=R,
                  initial_state_mean=mu_0,
                  initial_state_covariance=Sigma_0,
                  em_vars=em_vars)

# %% aplico como en el ejemplo
kf = kf.em(measurementsX)
# imprimo matrices corregidas por EM
kf.transition_matrices
kf.observation_matrices
kf.em_vars
kf.transition_covariance
kf.observation_covariance

(filtered_state_means, filtered_state_covariances) = kf.filter(measurementsX)
# (smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurementsX)

# %% plot
x_error = np.sqrt(filtered_state_covariances[:,0,0])
r_error = np.sqrt(filtered_state_covariances[:,1,1])
ordenadas = np.arange(len(x_error))

plt.figure()
plt.errorbar(ordenadas,filtered_state_means[:,0],yerr=x_error,label="x")
plt.errorbar(ordenadas,filtered_state_means[:,1],yerr=r_error,label="r_x")
plt.plot(filtered_state_means[:,0],label="x")
plt.plot(filtered_state_means[:,1],label="r_x")
plt.plot(measurementsX,'k',label="datos crudos")
plt.legend(loc='best')

'''
la covarianza de EM en relidad no cambia muchorespecto a lo dado inicialmente.
no es para sorepnderse ya que el filtro no tiene info suficiente.
'''
