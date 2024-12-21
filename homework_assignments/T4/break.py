import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm



def simulate_process(alpha, beta, phi, y0, n_periods):
    epsilon = np.random.normal(0, 1, n_periods)  # Gaussian white noise
    # epsilon = np.random.normal(0, 10, n_periods)
    y = np.zeros(n_periods)
    y[0] = y0
    for t in range(1, n_periods):
        if t != int(0.66 * n_periods):
            y[t] = alpha + beta * t + phi * y[t - 1] + epsilon[t]
        else:
            y[t] = alpha + beta * t + phi * y[t - 1] + epsilon[t] + 30
    return y



# Parameters
alpha = 1
beta = 0
phi = 1
y0 = 0
n_simulations = 1000  # Number of Monte Carlo simulations
n_periods = 100  # Number of time periods


T_break = int(0.66 * n_periods)  # T_break


time = np.arange(n_periods) # t = 0, 1, ..., n_periods-1

D1 = np.where(time[:-1] != T_break, 0, 1)
D2 = np.where(time[:-1] < (T_break - 1), 0, 1)

# print(f"T_break: {T_break}")
# print("D1:", D1)
# print("D2:", D2)




# Monte Carlo simulations

simulations_break = np.zeros((n_simulations, n_periods))

np.random.seed(10)  # For reproducibility / comparabilidad
for i in range(n_simulations):
    # Problema: tendría que encontrar los parámetros de la transformación
    simulations_break[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    # print(simulations_break[i])

# Plot the results

plt.figure(figsize=(12, 6))
plt.plot(simulations_break[i])
plt.title(r'Monte Carlo Simulation 1 of Random Walk w/Drif w/Break')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

plt.savefig("break_sim_1.svg", format='svg')

plt.show()


# All together
plt.figure(figsize=(12, 6))
for sim in simulations_break:
    plt.plot(sim)
plt.title(r'Monte Carlo Simulation of Random Walk w/Drif w/Break')
plt.xlabel('Time')
plt.ylabel(r"$y_t$")
plt.grid()

plt.savefig("monte_carlo_break.svg", format='svg')

plt.show()

# Need to regress Yt on:

# a constant
# Lag
# t
# D1
# D2

statistics_break = np.zeros(n_simulations)
for i in range(n_simulations):
    y = simulations_break[i]              
    y_lag = y[:-1]                            
    t = time[:-1]                             
    delta_y_lag = np.diff(y_lag, prepend=0)  # First difference of y_lag (Δy_{t-1})

    X = np.column_stack((np.ones(len(y_lag)),
                         y_lag,               
                         t,                   
                         D1,                  
                         D2,                  
                         delta_y_lag))        # Δy_{t-1}

    model = sm.OLS(y[1:], X).fit()

    statistics_break[i] = model.tvalues[1]  #  1 : coefficient y_lag


print("Plotting Distributions...")

plt.figure(figsize=(10, 6))

# Plot t-statistics
plt.hist(statistics_break, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{a}_1,\ T=100$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("test_perron_T_100.svg", format='svg')

plt.show()


critical_values_mode_T_100 = np.percentile(statistics_break, [5, 10])
print(f"Valor crítico al 90% Modelo 1: {critical_values_mode_T_100[0]}")
print(f"Valor crítico al 95% Modelo 1: {critical_values_mode_T_100[1]}")


n_periods = 396  # Number of time periods


T_break = int(0.66 * n_periods)  # T_break


time = np.arange(n_periods) # t = 0, 1, ..., n_periods-1

D1 = np.where(time[:-1] < T_break, 0, 1)
D2 = np.where(time[:-1] < (T_break - 1), 0, 1)

# print(f"T_break: {T_break}")
# print("D1:", D1)
# print("D2:", D2)




# Monte Carlo simulations

simulations_break = np.zeros((n_simulations, n_periods))

np.random.seed(10)  # For reproducibility / comparabilidad
for i in range(n_simulations):
    # Problema: tendría que encontrar los parámetros de la transformación
    simulations_break[i] = simulate_process(alpha, beta, phi, y0, n_periods)
    # print(simulations_break[i])

# Need to regress Yt on:

# a constant
# Lag
# t
# D1
# D2

statistics_break = np.zeros(n_simulations)
for i in range(n_simulations):
    y = simulations_break[i]              
    y_lag = y[:-1]                            
    t = time[:-1]                             
    delta_y_lag = np.diff(y_lag, prepend=0)  # First difference of y_lag (Δy_{t-1})

    X = np.column_stack((np.ones(len(y_lag)),
                         y_lag,               
                         t,                   
                         D1,                  
                         D2,                  
                         delta_y_lag))        # Δy_{t-1}

    model = sm.OLS(y[1:], X).fit()

    statistics_break[i] = model.tvalues[1]  #  1 : coefficient y_lag


print("Plotting Distributions...")

plt.figure(figsize=(10, 6))

# Plot t-statistics
plt.hist(statistics_break, bins=30, color='black', edgecolor='black', density=True)
plt.title(r"Distribución Estadístico t con $\hat{a}_1,\ T=396$")
plt.xlabel('t-value')
plt.ylabel('Prob.')

plt.savefig("test_perron_T_396.svg", format='svg')

plt.show()


critical_values_mode_T_396 = np.percentile(statistics_break, [5, 10])
print(f"Valor crítico al 90% Modelo 2: {critical_values_mode_T_396[0]}")
print(f"Valor crítico al 95% Modelo 2: {critical_values_mode_T_396[1]}")
