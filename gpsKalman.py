
# %%
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt

# %% example from pykalman page http://pykalman.github.io/
kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]],
                  observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])

measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)




# %% load gps data
dataFile = "/home/sebalander/Code/bayesian/data/20161006224131.txt"

data = np.loadtxt(dataFile,delimiter=",",skiprows=50,usecols=(1,2)).T
# pongo los datos medidos respecto a la media para que quede como referencia
# multiplico todo por 1e5 para que quede en nros mas manejables
data = (data.T - np.mean(data,axis=1)).T * 1e6

# tomo solo las coordenadas x como medicion
measurementsX = data[0]

plt.plot(data[0],data[1])
plt.plot(measurementsX)


# %% parametros kalman
# transition matrix
A = np.eye(2)
# measurement matrix
H = np.ones(2)
# the vector state must be [x, rx], the position and the gps error
# transition covariance
q = np.cov(measurementsX) # covariance of all data
f = 1e-1 # weigth associated to actual position (very small)
g = np.sqrt(1-f**2)
Q = q*np.array([[f,0],[0,g]])
# observation covariance: zero because we measure the state of noise as it is?
R = 0
# initial state mean
mu_0 = [-20, 0]
# initial state covariance
Sigma_0 = 40**2 * np.eye(2,2)
# que cosas optimizar por EM
# ['transition_matrices',     'observation_matrices', 'transition_offsets',
# 'observation_offsets',     'transition_covariance', 'observation_covariance',
# 'initial_state_mean',     'initial_state_covariance'] or 'all'
em_vars = ['transition_covariance', 'observation_covariance',
           'initial_state_mean',     'initial_state_covariance'] # "all"

# %% defino el filtro
kf = KalmanFilter(transition_matrices=A,
                  observation_matrices=H,
                  transition_covariance=Q,
                  observation_covariance=R,
                  initial_state_mean=mu_0,
                  initial_state_covariance=Sigma_0,
                  em_vars=em_vars)

# %% filtro son ajustar nada

(filtered_state_means, filtered_state_covariances) = kf.filter(measurementsX)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurementsX)

plt.figure()
plt.plot(measurementsX,label="latitud medida")
plt.plot(filtered_state_means[:,0],label="posicion estimada")
plt.plot(filtered_state_means[:,1],label="error GPS")
plt.legend()
plt.ylabel("latitud")

# %% aplico como en el ejemplo
kf = kf.em(measurementsX)
# imprimo matrices corregidas por EM
kf.transition_covariance
kf.observation_covariance

(filtered_state_means, filtered_state_covariances) = kf.filter(measurementsX)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurementsX)

# %%
plt.figure()
plt.plot(measurementsX,label="latitud medida")
plt.plot(filtered_state_means[:,0],label="posicion estimada")
plt.plot(filtered_state_means[:,1],label="error GPS")
plt.legend()
plt.ylabel("latitud")
