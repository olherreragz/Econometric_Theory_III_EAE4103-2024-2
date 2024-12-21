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


length_aic_ar1 = pd.read_csv('length_aic_ar1.csv')
length_bic_ar1 = pd.read_csv('length_bic_ar1.csv')
length_aic_ar2 = pd.read_csv('length_aic_ar2.csv')
length_bic_ar2 = pd.read_csv('length_bic_ar2.csv')


length_aic_ar2.columns = ['length']
length_bic_ar2.columns = ['length']
length_aic_ar1.columns = ['length']
length_bic_ar1.columns = ['length']


# print(length_aic_ar1)
# print(length_bic_ar1)
# print(length_aic_ar2)
# print(length_aic_ar2)


df_recm_aic_inflation = pd.read_csv('df_recm_aic_inflation.csv')
df_recm_bic_inflation = pd.read_csv('df_recm_bic_inflation.csv')
df_recm_aic_spot = pd.read_csv('df_recm_aic_spot.csv')
df_recm_bic_spot = pd.read_csv('df_recm_bic_spot.csv')


# print(df_recm_aic_inflation)
# print(df_recm_bic_inflation)
# print(df_recm_aic_spot)
# print(df_recm_bic_spot )


# DF 1

length_aic_ar2 = pd.DataFrame(length_aic_ar2, columns=['length'])
recms_aic_inflation = pd.DataFrame(df_recm_aic_inflation, columns=['recm'])

df_recm_aic_inflation = pd.concat([length_aic_ar2, recms_aic_inflation], axis=1)
df_recm_aic_inflation.columns = ['length', 'recm']

# DF 2

length_bic_ar2 = pd.DataFrame(length_bic_ar2, columns=['length'])
recms_bic_inflation = pd.DataFrame(recms_aic_inflation, columns=['recm'])

df_recm_bic_inflation = pd.concat([length_bic_ar2, recms_bic_inflation], axis=1)
df_recm_bic_inflation.columns = ['length', 'recm']


# DF 3

length_aic_ar1 = pd.DataFrame(length_aic_ar1, columns=['length'])
recms_aic_spot = pd.DataFrame(df_recm_aic_spot, columns=['recm'])

df_recm_aic_spot = pd.concat([length_aic_ar1, recms_aic_spot], axis=1)
df_recm_aic_spot.columns = ['length', 'recm']

# DF 4

length_bic_ar1 = pd.DataFrame(length_bic_ar1, columns=['length'])
recms_bic_spot = pd.DataFrame(df_recm_bic_spot, columns=['recm'])

df_recm_bic_spot = pd.concat([length_bic_ar1, recms_bic_spot], axis=1)
df_recm_bic_spot.columns = ['length', 'recm']


print("Dataframe with RECMs AIC for Inflation:\n")
print(df_recm_aic_inflation)

print("Dataframe with RECMs BIC for Inflation:\n")
print(df_recm_bic_inflation)

print("Dataframe with RECMs AIC for Spot:\n")
print(df_recm_aic_spot)

print("Dataframe with RECMs BIC for Spot:\n")
print(df_recm_bic_spot)


avg_aic_inflation = df_recm_aic_inflation.groupby('length')['recm'].mean().reset_index()
avg_bic_inflation = df_recm_bic_inflation.groupby('length')['recm'].mean().reset_index()
avg_aic_spot = df_recm_aic_spot.groupby('length')['recm'].mean().reset_index()
avg_bic_spot = df_recm_bic_spot.groupby('length')['recm'].mean().reset_index()

avg_aic_inflation.rename(columns={'recm': 'average_recm'}, inplace=True)
avg_bic_inflation.rename(columns={'recm': 'average_recm'}, inplace=True)
avg_aic_spot.rename(columns={'recm': 'average_recm'}, inplace=True)
avg_bic_spot.rename(columns={'recm': 'average_recm'}, inplace=True)


print("Dataframe 1 Group by Inflation:\n")
print(avg_aic_inflation)

print("Dataframe 2 Group by Inflation:\n")
print(avg_bic_inflation)

print("Dataframe 1 Group by Spot:\n")
print(avg_aic_spot)

print("Dataframe 2 Group by Spot:\n")
print(avg_bic_spot)


# fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

marker_size = 8
line_width = 2.5

# Subplot 1: Inflation (AIC and BIC)
axes[0].plot(avg_aic_inflation['length'], avg_aic_inflation['average_recm'], 
             label='AIC', marker='o', markersize=marker_size, linestyle='-', linewidth=line_width, color='gray')
axes[0].plot(avg_bic_inflation['length'], avg_bic_inflation['average_recm'], 
             label='BIC', marker='^', markersize=marker_size, linestyle='-', linewidth=line_width, color='black')

axes[0].set_title(r'Inflation RECMs', fontsize=14)
axes[0].set_xlabel('Coeficientes', fontsize=12)
axes[0].set_ylabel('Cuociente RECM', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend(loc='upper right', fontsize=10)

# Subplot 2: Exchange Rate (AIC and BIC)
axes[1].plot(avg_aic_spot['length'], avg_aic_spot['average_recm'], 
             label='AIC', marker='o', markersize=marker_size, linestyle='-', linewidth=line_width, color='gray')
axes[1].plot(avg_bic_spot['length'], avg_bic_spot['average_recm'], 
             label='BIC', marker='^', markersize=marker_size, linestyle='-', linewidth=line_width, color='black')

axes[1].set_title(r'Exchange Rate RECMs', fontsize=14)
axes[1].set_xlabel('Coeficientes', fontsize=12)
axes[1].set_ylabel('Cuociente RECM', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend(loc='upper right', fontsize=10)

plt.tight_layout()

plt.savefig("recm_by_order.svg", format='svg')

plt.show()


