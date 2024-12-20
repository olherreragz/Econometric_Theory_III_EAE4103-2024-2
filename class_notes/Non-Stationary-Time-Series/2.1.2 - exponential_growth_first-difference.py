"""
La única diferencia de este código con `2.0.2 - stochastic_trend_(first-difference_of_slide_6).py`
es que en parámetros de Process 3 MA(\infty)
el rho usado es 2 en vez de 1

"""


import numpy as np
import matplotlib.pyplot as plt


def simulate_process_ma_infty_first_diff(alpha, beta, phi, y0, n_periods):
    epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    # epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        phi_powers = [phi**i for i in range(t)]
        dynamic_lag_poly_MA = sum(phi_powers[i] * epsilon[t - 1 - i] for i in range(t))
        y[t] = alpha + beta * t + dynamic_lag_poly_MA

    y_diff = np.diff(y, prepend=y[0])  # First difference with padding
    return y_diff

# Parameters
alpha = 1
beta = 2
phi = 1
y0 = 1
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_ma_first_difference = np.zeros((n_simulations, n_periods))

np.random.seed(10)  # For reproducibility / comparabilidad
for i in range(n_simulations):
    simulations_ma_first_difference[i] = simulate_process_ma_infty_first_diff(alpha, beta, phi, y0, n_periods)
    print(simulations_ma_first_difference[i])


# Average and sample paths
mean_first_difference_simulation_ma = simulations_ma_first_difference.mean(axis=0)
sample_first_difference_simulation_ma = simulations_ma_first_difference[0]

plt.figure(figsize=(12, 6))
plt.plot(mean_first_difference_simulation_ma, label=r"Mean Simulation $\Delta y_t$", linestyle='--', linewidth=2)
plt.plot(sample_first_difference_simulation_ma, label=r"Sample $\Delta y_t$")
plt.title(r"First Difference of $y_t$")
plt.xlabel('Time')
plt.ylabel(r"$\Delta y_t$")
plt.grid()

plt.tight_layout()  # Adjust spacing
plt.show()


plt.figure(figsize=(12, 6))
for sim in simulations_ma_first_difference:
    plt.plot(sim)
plt.title(r"Monte Carlo Simulations of $\Delta y_t$")
plt.xlabel('Time')
plt.ylabel(r"$\Delta y_t$")
plt.grid()

plt.tight_layout()  # Adjust spacing
plt.show()



