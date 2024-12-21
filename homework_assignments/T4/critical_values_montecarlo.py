import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def simulate_process(alpha, beta, phi, y0, n_periods):
    # epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        y[t] = alpha + beta * t + phi * y[t - 1] + epsilon[t]
    return y

def compute_t_statistic(y):
    dy = np.diff(y)
    y_lag = y[:-1]

    X = sm.add_constant(y_lag)
    model = sm.OLS(dy, X).fit()
    t_stat = model.tvalues[1]
    return t_stat


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
phi = 1
y0 = 0
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_NO_ARMA_T_100 = np.zeros((n_simulations, n_periods))

np.random.seed(100)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA_T_100[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_T_100[i])

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_NO_ARMA_T_100:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Random Walk $y_t= y_{t-1} + \epsilon_t,\ T=100$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

plt.savefig("monte_carlo_model_1.svg", format='svg')

plt.show()


n_periods = 300  # Number of time periods

simulations_NO_ARMA_T_300 = np.zeros((n_simulations, n_periods))

for i in range(n_simulations):
    simulations_NO_ARMA_T_300 [i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_T_300 [i])


t_statistics_model_1_T_100 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_T_100[i]
    t_statistics_model_1_T_100[i] = compute_t_statistic_no_const(y_simulated)

t_statistics_model_1_T_300 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_T_300[i]
    t_statistics_model_1_T_300[i] = compute_t_statistic_no_const(y_simulated)



print("Plotting Distributions...")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

# Plot t-statistics
plt.hist(t_statistics_model_1_T_100, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 1,\ T=100$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.subplot(1, 2, 2)

# Plot t-statistics
plt.hist(t_statistics_model_1_T_300, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 1,\ T=300$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("T_stat_model_1.svg", format='svg')

plt.show()


""" Process 2 """

# Parameters
alpha = 1
beta = 0
phi = 1
y0 = 0
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_NO_ARMA_rwd_T_100 = np.zeros((n_simulations, n_periods))

np.random.seed(200)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA_rwd_T_100[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_rwd_T_100[i])

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_NO_ARMA_rwd_T_100:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Random Walk $y_t = \alpha + \beta t + (\phi + 1) y_{t-1} + \epsilon_,\ T=100$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

plt.savefig("monte_carlo_model_2.svg", format='svg')

plt.show()


n_periods = 300  # Number of time periods

simulations_NO_ARMA_rwd_T_300 = np.zeros((n_simulations, n_periods))

for i in range(n_simulations):
    simulations_NO_ARMA_rwd_T_300[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_rwd_T_300[i])


t_statistics_model_2_T_100 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_rwd_T_100[i]
    t_statistics_model_2_T_100[i] = compute_t_statistic(y_simulated)

t_statistics_model_2_T_300 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_rwd_T_300[i]
    t_statistics_model_2_T_300[i] = compute_t_statistic(y_simulated)


print("Plotting Distributions...")

# Plot t-statistics
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

plt.hist(t_statistics_model_2_T_100, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 2,\ T=100$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.subplot(1, 2, 2)

# Plot t-statistics
plt.hist(t_statistics_model_2_T_300, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 2,\ T=300$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("T_stat_model_2.svg", format='svg')

plt.show()


""" Process 3 """

# Parameters
alpha = 1
beta = 1
phi = 1
y0 = 0
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


# Monte Carlo simulations

simulations_NO_ARMA_rw_w_trend_T_100 = np.zeros((n_simulations, n_periods))

np.random.seed(300)  # For reproducibility
for i in range(n_simulations):
    simulations_NO_ARMA_rw_w_trend_T_100[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_rw_w_trend_T_100[i])

# All together
plt.figure(figsize=(12, 6))
for sim in simulations_NO_ARMA_rw_w_trend_T_100:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Random Walk $y_t = \alpha + \beta t + y_{t-1} + \epsilon_t,\ T=100$ ')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

plt.savefig("monte_carlo_model_3.svg", format='svg')

plt.show()


n_periods = 300  # Number of time periods

simulations_NO_ARMA_rw_w_trend_T_300 = np.zeros((n_simulations, n_periods))

for i in range(n_simulations):
    simulations_NO_ARMA_rw_w_trend_T_300[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    print(simulations_NO_ARMA_rw_w_trend_T_300[i])


t_statistics_model_3_T_100 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_rw_w_trend_T_100[i]
    t_statistics_model_3_T_100[i] = compute_t_statistic(y_simulated)


t_statistics_model_3_T_300 = np.zeros(n_simulations)

for i in range(n_simulations):
    y_simulated = simulations_NO_ARMA_rw_w_trend_T_300[i]
    t_statistics_model_3_T_300[i] = compute_t_statistic(y_simulated)



print("Plotting Distributions...")

# Plot t-statistics
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)

plt.hist(t_statistics_model_3_T_100, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 3,\ T=100$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.subplot(1, 2, 2)

# Plot t-statistics
plt.hist(t_statistics_model_3_T_300, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{\phi}, Modelo\ 3,\ T=300$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("T_stat_model_3.svg", format='svg')

plt.show()



print("T=100\n")

critical_values_model_1_T_100 = np.percentile(t_statistics_model_1_T_100, [5, 10])
print(f"Valor crítico al 90% Modelo 1: {critical_values_model_1_T_100[0]}")
print(f"Valor crítico al 95% Modelo 1: {critical_values_model_1_T_100[1]}")

critical_values_model_2_T_100 = np.percentile(t_statistics_model_2_T_100, [5, 10])
print(f"Valor crítico al 90% Modelo 1: {critical_values_model_2_T_100[0]}")
print(f"Valor crítico al 95% Modelo 1: {critical_values_model_2_T_100[1]}")

critical_values_model_3_T_100 = np.percentile(t_statistics_model_3_T_100, [5, 10])
print(f"Valor crítico al 90% Modelo 3: {critical_values_model_3_T_100[0]}")
print(f"Valor crítico al 95% Modelo 3: {critical_values_model_3_T_100[1]}")


print("T=300\n")

critical_values_model_1_T_300 = np.percentile(t_statistics_model_1_T_300, [5, 10])
print(f"Valor crítico al 90% Modelo 1: {critical_values_model_1_T_300[0]}")
print(f"Valor crítico al 95% Modelo 1: {critical_values_model_1_T_300[1]}")

critical_values_model_2_T_300 = np.percentile(t_statistics_model_2_T_300, [5, 10])
print(f"Valor crítico al 90% Modelo 1: {critical_values_model_2_T_300[0]}")
print(f"Valor crítico al 95% Modelo 1: {critical_values_model_2_T_300[1]}")

critical_values_model_3_T_300 = np.percentile(t_statistics_model_3_T_300, [5, 10])
print(f"Valor crítico al 90% Modelo 3: {critical_values_model_3_T_300[0]}")
print(f"Valor crítico al 95% Modelo 3: {critical_values_model_3_T_300[1]}")
