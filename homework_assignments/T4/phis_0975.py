import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def simulate_process(alpha, beta, phi, y0, n_periods):
    epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    # epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        y[t] = alpha + beta * t + phi * y[t - 1] + epsilon[t]
    return y

def compute_t_statistic_no_const(y):
    dy = np.diff(y)
    y_lag = y[:-1]

    model = sm.OLS(dy, y_lag).fit()
    t_stat = model.tvalues[0]
    return t_stat


""" Process 1 """

# Parameters
alpha = 0
beta = 0
phi = 1.975
y0 = 0
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_NO_ARMA_Y = np.zeros((n_simulations, n_periods))

np.random.seed(500)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA_Y[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    # print(simulations_NO_ARMA_Y[i])

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_NO_ARMA_Y:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of $Y_t$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

# plt.savefig("monte_carlo_model_Y.svg", format='svg')

plt.show()


simulations_NO_ARMA_Z = np.zeros((n_simulations, n_periods))

np.random.seed(1000)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA_Z [i] = simulate_process(alpha, beta, phi, y0, n_periods)
    # print(simulations_NO_ARMA_Z [i])


t_statistics_model_1_Y = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_Y[i]
    t_statistics_model_1_Y[i] = compute_t_statistic_no_const(y_simulated)


t_statistics_model_1_Z = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_Z[i]
    t_statistics_model_1_Z[i] = compute_t_statistic_no_const(y_simulated)


plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

# Plot t-statistics
plt.hist(t_statistics_model_1_Y, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi_Y}, Modelo\ Y_t$", pad=20)
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.subplot(1, 2, 2)

# Plot t-statistics
plt.hist(t_statistics_model_1_Z, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi_Z}, Modelo\ Z_t$", pad=20)
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("T_stat_models_Y_and_Z.svg", format='svg')

plt.show()

print("sim Y")
print(simulations_NO_ARMA_Y)

delta_Y = np.diff(simulations_NO_ARMA_Y, axis=1)  # First difference of Y
delta_Z = np.diff(simulations_NO_ARMA_Z, axis=1)  # First difference of Z

# print("Delta Y")
# print(delta_Y)
# print("Delta Y")
# print(delta_Z)

# Regression: Delta_Y vs Delta_Z (without constant)
t_statistics_second_stage = np.zeros(n_simulations)
for i in range(n_simulations):
    model = sm.OLS(delta_Y[i], delta_Z[i]).fit()  # Regr w/ constant
    t_statistics_second_stage[i] = model.tvalues[0]  # t-statistic for delta_Z coefficient

# Plot t-statistic
plt.figure(figsize=(10, 6))
plt.hist(t_statistics_second_stage, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t $\gamma$ (Segunda Etapa: $\Delta Y_t$ vs $\Delta Z_t$)")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("T_stat_model_Y_on_Z.svg", format='svg')

plt.show()


