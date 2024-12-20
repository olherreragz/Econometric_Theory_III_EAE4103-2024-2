import numpy as np
import matplotlib.pyplot as plt


# Function to simulate the process
# Problema: tendría que encontrar los parámetros de la transformación hacia el AR(1)
# para que sea equivalente al MA(infty)
def simulate_process(alpha, beta, phi, y0, n_periods):
    # epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        y[t] = alpha + beta * t + phi * y[t - 1] + epsilon[t]
    return y


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


""" Process 1 """

# Parameters
alpha = 1
beta = 2
phi = 0
y0 = 1
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods



# Monte Carlo simulations

simulations_NO_ARMA = np.zeros((n_simulations, n_periods))

np.random.seed(10)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA[i])

# Plot the results

# Average and sample paths
mean_simulation_NO_ARMA = simulations_NO_ARMA.mean(axis=0)
sample_simulation_NO_ARMA = simulations_NO_ARMA[0]

plt.figure(figsize=(12, 6))
plt.plot(mean_simulation_NO_ARMA, label='Mean Simulation', linestyle='--', linewidth=2)
plt.plot(sample_simulation_NO_ARMA, label='Sample Path')
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.legend()
plt.grid()
plt.show()

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_NO_ARMA:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()
plt.show()




""" Process 2 """

# Parameters
alpha = 1
beta = 2
phi = 0.5
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations = np.zeros((n_simulations, n_periods))


np.random.seed(10)  # For reproducibility / comparabilidad
for i in range(n_simulations):
    simulations[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations[i])

# Plot the results

# Average and sample paths
mean_simulation = simulations.mean(axis=0)
sample_simulation = simulations[0]

plt.figure(figsize=(12, 6))
plt.plot(mean_simulation, label='Mean Simulation', linestyle='--', linewidth=2)
plt.plot(sample_simulation, label='Sample Path')
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.legend()
plt.grid()
plt.show()

# All together
plt.figure(figsize=(12, 6))
for sim in simulations:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()
plt.show()


""" Comparison """


plt.figure(figsize=(12, 6))
plt.plot(mean_simulation_NO_ARMA, label=r"Mean Simulation $y_t= \alpha + \beta t + \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(mean_simulation, label=r"Mean Simulation $y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(sample_simulation_NO_ARMA, label=r"$y_t= \alpha + \beta t + \epsilon_t$")
plt.plot(sample_simulation, label=r"$y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$")
plt.title(r'Monte Carlo Simulation of Non-Stationary Process $y_t= \alpha + \beta t + \epsilon_t$ & $y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.legend()
plt.grid()
plt.show()


all_simulations = np.concatenate((simulations_NO_ARMA, simulations))
y_min = all_simulations.min()
y_max = all_simulations.max()

plt.subplots(figsize=(12, 6))

plt.subplot(1, 2, 1)
for sim in simulations_NO_ARMA:
    plt.plot(sim)
plt.title(r'$y_t= \alpha + \beta t + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.subplot(1, 2, 2)
for sim in simulations:
    plt.plot(sim)
plt.title(r'$y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.tight_layout()  # Adjust spacing
plt.show()


""" Process 3 """


""" MA(\infty) == AR(1)? """
# Problema: tendría que encontrar los parámetros de la transformación hacia el AR(1)
# para que sea equivalente al MA(infty)


# Parameters
alpha = 1
beta = 2
phi = 0.5
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

# Average and sample paths
mean_simulation_ma = simulations_ma.mean(axis=0)
sample_simulation_ma = simulations_ma[0]


all_simulations = np.concatenate((simulations, simulations_ma))  # `simulations`` corresponde al AR(1)
y_min = all_simulations.min()
y_max = all_simulations.max()

plt.subplots(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mean_simulation, label=r"Mean Simulation $y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(sample_simulation, label=r"$y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$")
plt.title(r"$y_t= \alpha + \beta t + \phi y_{t-1} + \epsilon_t$")
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(mean_simulation_ma, label=r"Mean Simulation $y_t= \alpha + \beta t + \theta (L) \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(sample_simulation_ma, label=r"$y_t= \alpha + \beta t + \theta (L) \epsilon_t$")
plt.title(r"$y_t= \alpha + \beta t + \theta (L) \epsilon_t$")
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.tight_layout()  # Adjust spacing
plt.show()


""" MA(\infty) vs MA(0) """


all_simulations = np.concatenate((simulations_NO_ARMA, simulations_ma))
y_min = all_simulations.min()
y_max = all_simulations.max()

plt.subplots(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(mean_simulation_NO_ARMA, label=r"Mean Simulation $y_t= \alpha + \beta t + \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(sample_simulation_NO_ARMA, label=r"$y_t= \alpha + \beta t + \epsilon_t$")
plt.title(r"$y_t= \alpha + \beta t + \epsilon_t$")
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(mean_simulation_ma, label=r"Mean Simulation $y_t= \alpha + \beta t + \theta (L) \epsilon_t$", linestyle='--', linewidth=2)
plt.plot(sample_simulation_ma, label=r"$y_t= \alpha + \beta t + \theta (L) \epsilon_t$")
plt.title(r"$y_t= \alpha + \beta t + \theta (L) \epsilon_t$")
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.ylim(y_min, y_max)  # Set y-axis limits
plt.grid()

plt.tight_layout()  # Adjust spacing
plt.show()


