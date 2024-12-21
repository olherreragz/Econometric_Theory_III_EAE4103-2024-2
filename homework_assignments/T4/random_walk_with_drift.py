import numpy as np
import matplotlib.pyplot as plt



def simulate_process_ma_infty(alpha, beta, phi, y0, n_periods):
    # epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        phi_powers = [phi**i for i in range(t)]
        dynamic_lag_poly_MA = sum(phi_powers[i] * epsilon[t - 1 - i] for i in range(t))
        y[t] = alpha + beta * t + dynamic_lag_poly_MA
    return y


""" MA(\infty) == AR(1)? """
# Problema: tendría que encontrar los parámetros de la transformación hacia el AR(1)
# para que sea equivalente al MA(infty)


# Parameters
alpha = 1
beta = 2
phi = 1
y0 = 1
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_ma = np.zeros((n_simulations, n_periods))

np.random.seed(10)  # For reproducibility / comparabilidad
for i in range(n_simulations):
    # Problema: tendría que encontrar los parámetros de la transformación
    simulations_ma[i] = simulate_process_ma_infty(alpha, beta, phi, y0, n_periods)
    print(simulations_ma[i])

# Plot the results

# Average and sample paths
mean_simulation_ma = simulations_ma.mean(axis=0)
sample_simulation_ma = simulations_ma[0]


plt.figure(figsize=(12, 6))
plt.plot(mean_simulation_ma, label='Mean Simulation', linestyle='--', linewidth=2)
plt.plot(sample_simulation_ma, label='Sample Path')
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \theta (L) \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.legend()
plt.grid()
plt.show()

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_ma:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \theta (L) \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()
plt.savefig("rw_p_wn.svg", format='svg')
plt.show()

