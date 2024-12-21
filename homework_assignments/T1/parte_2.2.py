import sys
import os
import pickle
import random
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.collections import PolyCollection
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns
from sklearn.metrics import mean_squared_error


def dynamic_forecast_with_out_of_sample(model_fit, simulation_data, n_lags, start_index, steps):
    """
    Performs dynamic out-of-sample forecasting.
    
    Parameters:
    - model_fit: The fitted AutoReg model.
    - simulation_data: The full dataset (including out-of-sample values).
    - start_index: The index from which to begin the dynamic forecast.
    - steps: Number of forecast steps to perform.
    
    Returns:
    - forecast: Array of dynamically predicted values.
    """

    forecast = []
    data = simulation_data[:start_index].copy()  # Use full dataset up to the forecast start point

    for step in range(steps):
        # print("checkpoint1")
        next_value = model_fit.predict(start=len(data) - n_lags, end=len(data) - n_lags)
        # print(next_value.iloc[0])
        # print("checkpoint2")
        forecast.append(next_value.iloc[0])  # Save the predicted value
        data = np.append(data, next_value)  # Update the data with the new prediction

    return np.array(forecast)


length_aic_ar1 = pd.read_csv('length_aic_ar1.csv')
print(length_aic_ar1)

length_bic_ar1 = pd.read_csv('length_bic_ar1.csv')
print(length_bic_ar1)

length_aic_ar2 = pd.read_csv('length_aic_ar2.csv')
print(length_aic_ar2)

length_bic_ar2 = pd.read_csv('length_bic_ar2.csv')
print(length_aic_ar2)


plt.figure(figsize=(15, 10))

plt.suptitle(r"Distribuciones de Órdenes por Criterios de Información, Inflación Japón", fontsize=16)

plt.subplot(1, 2, 1)
sns.histplot(length_aic_ar2['length_aic_ar2'], bins=30, kde=True, color='#a5d7ff', stat="probability", linewidth=0)
# kde_kws={"color": "darkblue"}
plt.title("Distribución de Rezagos por AIC")
plt.xlabel(r"$p$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplot(1, 2, 2)
sns.histplot(length_bic_ar2['length_bic_ar2'], bins=30, kde=True, color='#a5d7ff', stat="probability", linewidth=0)
plt.title("Distribución de Rezagos por BIC")
plt.xlabel(r"$p$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplots_adjust(hspace=0.3)

plt.savefig("orders_ic_inflation.svg", format='svg')

plt.show()



plt.figure(figsize=(15, 10))

plt.suptitle(r"Distribuciones de Órdenes por Criterios de Información, Retornos Yen", fontsize=16)

plt.subplot(1, 2, 1)
sns.histplot(length_aic_ar2['length_aic_ar2'], bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
# kde_kws={"color": "darkblue"}
plt.title("Distribución de Rezagos por AIC")
plt.xlabel(r"$p$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplot(1, 2, 2)
sns.histplot(length_bic_ar2['length_bic_ar2'], bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
plt.title("Distribución de Rezagos por BIC")
plt.xlabel(r"$p$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplots_adjust(hspace=0.3)

plt.savefig("orders_ic_spot_returns.svg", format='svg')

plt.show()


print("Dataframe simulaciones AR(2) training data:\n")
monte_carlo_simulations_ar2 = pd.read_csv('simulations_ar2.csv')
print(monte_carlo_simulations_ar2)


print("Dataframe simulaciones AR(1) training data:\n")
monte_carlo_simulations_ar1 = pd.read_csv('simulations_ar1.csv')
print(monte_carlo_simulations_ar1)


print("Dataframe simulaciones AR(2) w/ validation data:\n")
monte_carlo_simulations_ar2_long_version = pd.read_csv('simulations_ar2_long_version.csv')
print(monte_carlo_simulations_ar2_long_version)


print("Dataframe simulaciones AR(1) w/ validation data:\n")
monte_carlo_simulations_ar1_long_version = pd.read_csv('simulations_ar1_long_version.csv')
print(monte_carlo_simulations_ar1_long_version)




print(range(1, len(monte_carlo_simulations_ar1_long_version.columns)))
print(monte_carlo_simulations_ar1_long_version)
print(monte_carlo_simulations_ar1_long_version.columns)

## TEST


data_to_plot = monte_carlo_simulations_ar1_long_version.iloc[:, 2]
print(data_to_plot[275:])


plt.figure(figsize=(12, 6))
plt.plot(range(275,len(data_to_plot)), data_to_plot[275:], label='Actual Data', linestyle='--', color='blue')
plt.title('Simulation 1 Exchange Rate Japan:')
plt.xlabel('Time Index')
plt.ylabel(r'$\Delta \hat{e}$')
plt.legend()
plt.show()


recms_aic_inflation = []
recms_bic_inflation = []



for i in range(1,len(monte_carlo_simulations_ar2_long_version.columns)): 

    print(monte_carlo_simulations_ar2_long_version.iloc[:, i])
    simulation_data = monte_carlo_simulations_ar2_long_version.iloc[:, i]
    
    # up to 2018 for training
    train_data = simulation_data[:len(monte_carlo_simulations_ar2)]
    out_of_sample_length = len(simulation_data) - len(train_data)

    # AIC Model
    model_aic = AutoReg(train_data, lags=int(length_aic_ar2.iloc[i-1]))
    model_aic_fit = model_aic.fit()
    # AIC Model Forecast
    forecast_aic = dynamic_forecast_with_out_of_sample(
        model_aic_fit, simulation_data, n_lags=int(length_aic_ar2.iloc[i-1]),
        start_index=len(train_data), steps=out_of_sample_length
    )
    # print("Out-of-sample dynamic prediction:")
    # print(forecast_aic)
    
    # BIC Model
    model_bic = AutoReg(train_data, lags=int(length_bic_ar2.iloc[i-1]))
    model_bic_fit = model_bic.fit()
    # BIC Model Forecast
    forecast_bic = dynamic_forecast_with_out_of_sample(
        model_bic_fit, simulation_data, n_lags=int(length_bic_ar2.iloc[i-1]),
        start_index=len(train_data), steps=out_of_sample_length
    )
    
    # RMSE Calculation
    actual_data = simulation_data[len(train_data):]
    rmse_aic = np.sqrt(mean_squared_error(actual_data, forecast_aic))
    rmse_bic = np.sqrt(mean_squared_error(actual_data, forecast_bic))
    
    recms_aic_inflation.append(rmse_aic)
    recms_bic_inflation.append(rmse_bic)

    if i == 1:
        # Hardcoded start
        start_plot = 800
        
        forecast_range = range(len(train_data), len(train_data) + len(forecast_aic))
        actual_range = range(len(train_data), len(train_data) + len(actual_data))
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_plot,len(simulation_data)), simulation_data[start_plot:], label='Actual Data', linestyle='--', color='blue')
        plt.axvline(x=len(train_data), color='grey', linestyle='--', label='Forecast Start')
        plt.plot(
            forecast_range, forecast_aic, label='AIC Forecast', marker='o', linestyle='-', color='#adadad'
        )
        plt.plot(
            forecast_range, forecast_bic, label='BIC Forecast', marker='x', linestyle='-', color='#273746'
        )
        plt.plot(
            actual_range, actual_data, label='Actual Out-of-Sample', linestyle='solid', color='red'
        )
        plt.title('Simulation 1 Inflation Japan: Actual vs Forecast (AIC & BIC)')
        plt.xlabel('Time Index')
        plt.ylabel(r'$\pi, \hat{\pi}$')
        plt.legend()
        plt.savefig("forecast_inflation.svg", format='svg')
        plt.show()

    print(f"Simulation {i + 1} - RMSE AIC: {rmse_aic:.4f}, RMSE BIC: {rmse_bic:.4f}")


print("RMSE AIC Inflation:\n")
print(recms_aic_inflation)

print("RMSE BIC Inflation:\n")
print(recms_bic_inflation)


recms_aic_spot = []
recms_bic_spot = []

print("monte_carlo_simulations_ar1_long_version\n")
print(monte_carlo_simulations_ar1_long_version)

for i in range(1, len(monte_carlo_simulations_ar1_long_version.columns)): 

    simulation_data = monte_carlo_simulations_ar1_long_version.iloc[:, i]

    # print("simulation_data\n")
    # print(simulation_data)
    
    # up to 2018 for training
    train_data = simulation_data[:len(monte_carlo_simulations_ar1)]
    out_of_sample_length = len(simulation_data) - len(train_data)

    # print("train_data\n")
    # print(train_data)

    # AIC Model
    model_aic = AutoReg(train_data, lags=int(length_aic_ar1.iloc[i-1]))
    model_aic_fit = model_aic.fit()
    # AIC Model Forecast
    forecast_aic = dynamic_forecast_with_out_of_sample(
        model_aic_fit, simulation_data, n_lags=int(length_aic_ar1.iloc[i-1]),
        start_index=len(train_data), steps=out_of_sample_length
    )
    # print("Out-of-sample dynamic prediction:")
    # print(forecast_aic)
    
    # BIC Model
    model_bic = AutoReg(train_data, lags=int(length_bic_ar1.iloc[i-1]))
    model_bic_fit = model_bic.fit()
    # BIC Model Forecast
    forecast_bic = dynamic_forecast_with_out_of_sample(
        model_bic_fit, simulation_data, n_lags=int(length_bic_ar1.iloc[i-1]),
        start_index=len(train_data), steps=out_of_sample_length
    )
    
    # RMSE Calculation
    actual_data = simulation_data[len(train_data):]
    rmse_aic = np.sqrt(mean_squared_error(actual_data, forecast_aic))
    rmse_bic = np.sqrt(mean_squared_error(actual_data, forecast_bic))
    
    recms_aic_spot.append(rmse_aic)
    recms_bic_spot.append(rmse_bic)

    if i == 1:
        # Hardcoded start
        start_plot = 275
        
        forecast_range = range(len(train_data), len(train_data) + len(forecast_aic))
        actual_range = range(len(train_data), len(train_data) + len(actual_data))
        plt.figure(figsize=(12, 6))
        plt.plot(range(start_plot,len(simulation_data)), simulation_data[start_plot:], label='Actual Data', linestyle='--', color='blue')
        plt.axvline(x=len(train_data), color='grey', linestyle='--', label='Forecast Start')
        plt.plot(
            forecast_range, forecast_aic, label='AIC Forecast', marker='o', linestyle='-', color='#adadad'
        )
        plt.plot(
            forecast_range, forecast_bic, label='BIC Forecast', marker='x', linestyle='-', color='#273746'
        )
        plt.plot(
            actual_range, actual_data, label='Actual Out-of-Sample', linestyle='solid', color='red'
        )
        plt.title('Simulation 1 Exchange Rate Japan: Actual vs Forecast (AIC & BIC)')
        plt.xlabel('Time Index')
        plt.ylabel(r'$\Delta, \Delta \hat{e}$')
        plt.legend()
        plt.savefig("forecast_spot.svg", format='svg')
        plt.show()

    print(f"Simulation {i + 1} - RMSE AIC: {rmse_aic:.4f}, RMSE BIC: {rmse_bic:.4f}")

# print("RMSE AIC Inflation:\n")
# print(recms_aic_spot)

# print("RMSE BIC Inflation:\n")
# print(recms_bic_spot)


recms_aic_inflation = pd.DataFrame(recms_aic_inflation, columns=['recm'])

recms_bic_inflation = pd.DataFrame(recms_aic_inflation, columns=['recm'])

recms_aic_spot = pd.DataFrame(recms_aic_spot, columns=['recm'])

recms_bic_spot = pd.DataFrame(recms_bic_spot, columns=['recm'])


print("Dataframe with RECMs AIC for Inflation:\n")

print(recms_aic_inflation)

print("Dataframe with RECMs BIC for Inflation:\n")

print(recms_bic_inflation)

print("Dataframe with RECMs AIC for Spot:\n")

print(recms_aic_spot)

print("Dataframe with RECMs BIC for Spot:\n")

print(recms_bic_spot)


recms_aic_inflation.to_csv('df_recm_aic_inflation.csv', index=False)
recms_bic_inflation.to_csv('df_recm_bic_inflation.csv', index=False)
recms_aic_spot.to_csv('df_recm_aic_spot.csv', index=False)
recms_bic_spot.to_csv('df_recm_bic_spot.csv', index=False)

print("Csvs generated.")

