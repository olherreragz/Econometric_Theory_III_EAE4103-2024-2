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


# Simular 1 secuencia AR(2)
def simulate_ar2(constant, phi_1, phi_2, T=100):
    errors = np.random.normal(0, 1, T)  # Generate white noise errors
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = constant + phi_1 * y[t-1] + phi_2 * y[t-2] + errors[t]
    return y


# Simular 1 secuencia AR(1)
def simulate_ar1(constant, phi_1, T=100):
    errors = np.random.normal(0, 1, T)  # Generate white noise errors
    y = np.zeros(T)
    for t in range(2, T):
        y[t] = constant + phi_1 * y[t-1] + + errors[t]
    return y


# Simular 1 secuencia ARMA(1,1)
def simulate_arma_1_1(constant, phi_1, theta_1, T=1000):
    errors = np.random.normal(0, 1, T)  # Generate white noise errors
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = constant + phi_1 * y[t-1] + theta_1 * errors[t-1] + errors[t] 
    return y


def monte_carlo_ar2(constant, phi_1, phi_2, T=100, N=1000):
    simulations = np.zeros((N, T))
    for i in range(N):
        simulations[i] = simulate_ar2(constant, phi_1, phi_2, T)
    return simulations


def monte_carlo_ar_1(constant, phi_1, T=100, N=1000):
    simulations = np.zeros((N, T))
    for i in range(N):
        simulations[i] = simulate_ar1(constant, phi_1, T)
    return simulations


def monte_carlo_arma_1_1(constant, phi_1, theta_1, T=100, N=1000):
    simulations = np.zeros((N, T)) 
    for i in range(N):
        simulations[i] = simulate_arma_1_1(constant, phi_1, theta_1, T)
    return simulations



if not os.path.exists('df_inflation_monthly.csv') or not os.path.exists('df_exchange_rate_index_monthly.csv'):

    target_script_dir = r'data\downloads\code_snippets'
    sys.path.append(target_script_dir)

    import python_snippet_japan_cpi_changes
    import python_snippet_japan_exchange_rate_index

    df_inflation_monthly = python_snippet_japan_cpi_changes.df
    df_exchange_rate_index_monthly = python_snippet_japan_exchange_rate_index.df

    # print("Japna's CPI changes DatFrame:")
    # print(df_inflation_monthly)

    # print("Japna's Nominal effective exchange rate Broad basket DatFrame:")
    # print(df_exchange_rate_index_monthly)

    df_inflation_monthly = df_inflation_monthly[['TIME_PERIOD', 'OBS_VALUE']]
    df_exchange_rate_index_monthly = df_exchange_rate_index_monthly[['TIME_PERIOD', 'OBS_VALUE']]

    print("Cleaned Japan CPI Changes DataFrame:")
    print(df_inflation_monthly)

    print("Cleaned Japan Nominal Effective Exchange Rate DataFrame:")
    print(df_exchange_rate_index_monthly)

    df_inflation_monthly.to_csv('df_inflation_monthly.csv', index=True)
    df_exchange_rate_index_monthly.to_csv('df_exchange_rate_index_monthly.csv', index=True)

    print("DataFrames saved as CSV files.")

    # print(os.getcwd())

else:

    df_inflation_monthly = pd.read_csv('df_inflation_monthly.csv')
    df_exchange_rate_index_monthly = pd.read_csv('df_exchange_rate_index_monthly.csv')

    print("Loaded Japan CPI Changes DataFrame from CSV:")
    print(df_inflation_monthly)

    print("Loaded Japan Nominal Effective Exchange Rate DataFrame from CSV:")
    print(df_exchange_rate_index_monthly)

    # print(os.getcwd())


plt.figure(figsize=(10, 6))
plt.plot(
    df_inflation_monthly['TIME_PERIOD'],
    df_inflation_monthly['OBS_VALUE'],
    label='CPI Changes',
    color='#ff082d'
)
plt.xlabel('Time')
plt.ylabel(r'CPI Change $\pi$')
plt.title(r'Monthly Inflation $\pi$ of Japan')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("japan_inflation.svg", format='svg')
plt.show()


plt.figure(figsize=(10, 6))
plt.plot(
    df_exchange_rate_index_monthly['TIME_PERIOD'],
    df_exchange_rate_index_monthly['OBS_VALUE'],
    label='Exchange Rate Index', color='#33FF57'
)
plt.xlabel('Time')
plt.ylabel('Exchange Rate Index')
plt.title('Japan\'s Nominal Effective Exchange Rate Index')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("japan_spot.svg", format='svg')
plt.show()


df_exchange_rate_index_monthly['Percentage_Change'] = df_exchange_rate_index_monthly['OBS_VALUE'].pct_change() * 100

print("Japan's Monthly Nominal Effective Exchange Rate Changes:")
print(df_exchange_rate_index_monthly[['TIME_PERIOD', 'Percentage_Change']])


plt.figure(figsize=(10, 6))
plt.plot(
    df_exchange_rate_index_monthly['TIME_PERIOD'],
    df_exchange_rate_index_monthly['Percentage_Change'],
    label='Exchange Rate Index', color='#33FF57'
)
plt.xlabel('Time')
plt.ylabel(r'Exchange Rate Depreciation $\Delta e$')
plt.title(r'Nominal Exchange Rate Depreciation $\Delta e$ of Japanese yen ($JPY$)')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("japan_depreciation.svg", format='svg')
plt.show()


hypothesis_tex = r"""
El diseño de las hipótesis del test está dado por:
\newline

\begin{center}
    $H_0$: La serie es de raíz unitaria, i.e., no es estacionaria. \\
    $H_1:$ La serie es estacionaria. \\
\end{center}
"""
filename_hipotesis = "hipotesis_unit_roots_tests.tex"

with open(filename_hipotesis, 'w', encoding='utf-8') as f_hypothesis:
    f_hypothesis.write(hypothesis_tex)
print(f"LaTeX generado: {filename_hipotesis}")


# (A Little hardcoded):

Serie = "Japan's Monthly Inflation"

print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df_inflation_monthly['OBS_VALUE'].dropna(), autolag='AIC')
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {format(result[1], 'f')}")
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f"   {key}: {value}")
if result[1] < 0.05:
    print(f"\"{Serie}\" is likely stationary (reject the null hypothesis).")
else:
    print(f"\"{Serie}\" is likely non-stationary (fail to reject the null hypothesis).")
print()

results_filename="tabla_adf_results_inflation.tex"

tex_content = r"""
\begin{center}
        \centering
        \begin{tabular}{cc}
            Augmented Dickey-Fuller Test for """ + Serie + r""" \\
            \midrule
            ADF Statistic & """ + f"{result[0]:.6f} \\\\\n" + r"""
            p-value & """ + f"{result[1]:.6f} \\\\\n" + r"""
            \bottomrule
        \end{tabular}
\end{center}
\vspace{10pt}

"""

with open(results_filename, 'w', encoding='utf-8') as tex_file:
    tex_file.write(tex_content)
    print(f"LaTeX generado: {results_filename}")
    print()


Serie = "Japan's Nominal Exchange Rate Index"


print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df_exchange_rate_index_monthly['OBS_VALUE'].dropna(), autolag='AIC')
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {format(result[1], 'f')}")
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f"   {key}: {value}")
if result[1] < 0.05:
    print(f"\"{Serie}\" is likely stationary (reject the null hypothesis).")
else:
    print(f"\"{Serie}\" is likely non-stationary (fail to reject the null hypothesis).")
print()

results_filename="tabla_adf_results_spot.tex"

tex_content = r"""
\begin{center}
        \centering
        \begin{tabular}{cc}
            Augmented Dickey-Fuller Test for """ + Serie + r""" \\
            \midrule
            ADF Statistic & """ + f"{result[0]:.6f} \\\\\n" + r"""
            p-value & """ + f"{result[1]:.6f} \\\\\n" + r"""
            \bottomrule
        \end{tabular}
\end{center}
\vspace{10pt}

"""

with open(results_filename, 'w', encoding='utf-8') as tex_file:
    tex_file.write(tex_content)
    print(f"LaTeX generado: {results_filename}")
    print()


Serie = "Japan's Nominal Exchange Rate Returns"


print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df_exchange_rate_index_monthly['Percentage_Change'].dropna(), autolag='AIC')
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {format(result[1], 'f')}")
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f"   {key}: {value}")
if result[1] < 0.05:
    print(f"\"{Serie}\" is likely stationary (reject the null hypothesis).")
else:
    print(f"\"{Serie}\" is likely non-stationary (fail to reject the null hypothesis).")
print()

results_filename="tabla_adf_results_spot_returns.tex"

tex_content = r"""
\begin{center}
        \centering
        \begin{tabular}{cc}
            Augmented Dickey-Fuller Test for """ + Serie + r""" \\
            \midrule
            ADF Statistic & """ + f"{result[0]:.6f} \\\\\n" + r"""
            p-value & """ + f"{result[1]:.6f} \\\\\n" + r"""
            \bottomrule
        \end{tabular}
\end{center}
\vspace{10pt}

"""

with open(results_filename, 'w', encoding='utf-8') as tex_file:
    tex_file.write(tex_content)
    print(f"LaTeX generado: {results_filename}")
    print()


""" ~~~~~ Autocorrelogramas ~~~~~ """


series = df_inflation_monthly['OBS_VALUE'].dropna()

# Compute ACF and PACF
acf_vals = acf(series, nlags=20, fft=True) 
# pacf_vals = pacf(series, nlags=20)
pacf_vals = pacf(series, nlags=20, method="ywm")


# Plot ACF
plt.subplots(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.stem(
#     range(len(acf_vals)),
#     acf_vals,
#     basefmt=" ",
#     use_line_collection=True,
#     linefmt="#ff082d",
#     markerfmt="o",  # Marker shape
# )
# plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
ax1 = plt.subplot(1, 2, 1)
plot_acf(
    series,
    lags=20,
    ax=ax1,
    color='black',
    markerfacecolor='#122bff',
    markeredgecolor='#122bff',
    vlines_kwargs={"colors": "#ff082d"}
)
for item in ax1.collections:
    # change the color of the confidence interval 
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")
plt.title(r'Autocorrelograma Total Inflación $\pi$ Japón', fontsize=18)
plt.xlabel("Lag",fontsize=17)  
plt.ylabel("ACF",fontsize=17)
plt.xticks(range(0, len(acf_vals), 2), fontsize=15)    # Eje X


# Plot PACF
# plt.subplot(1, 2, 2)
# plt.stem(
#     range(len(pacf_vals)),
#     pacf_vals,
#     basefmt=" ",
#     use_line_collection=True,
#     linefmt="#ff082d",
#     markerfmt="o",  # Marker shape
# )
# plt.gca().get_children()[1].set_color("#122bff")  # Color markers
ax1 = plt.subplot(1, 2, 2)
plot_pacf(
    series,
    lags=20,
    ax=ax1,
    color='black',
    markerfacecolor='#122bff',
    markeredgecolor='#122bff',
    vlines_kwargs={"colors": "#ff082d"}
)
for item in ax1.collections:
    # change the color of the confidence interval 
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")

plt.title(r'Autocorrelograma Parcial Inflación $\pi$ Japón', fontsize=18)
plt.xlabel("Lag",fontsize=17)  
plt.ylabel("PACF",fontsize=17)
plt.xticks(range(0, len(acf_vals), 2), fontsize=15)    # Eje X

plt.tight_layout()
plt.savefig("acf_pacf_plot_inflation.svg", format='svg')
plt.show()


series = df_exchange_rate_index_monthly['Percentage_Change'].dropna()

# Compute ACF and PACF
acf_vals = acf(series, nlags=20, fft=True) 
# pacf_vals = pacf(series, nlags=20)
pacf_vals = pacf(series, nlags=20, method="ywm")


# Plot ACF
plt.figure(figsize=(14, 6))
# plt.subplot(1, 2, 1)
# plt.stem(
#     range(len(acf_vals)),
#     acf_vals,
#     basefmt=" ",
#     use_line_collection=True,
#     linefmt="#33FF57",
#     markerfmt="o",  # Marker shape
# )
# plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
ax1 = plt.subplot(1, 2, 1)
plot_acf(
    series,
    lags=20,
    ax=ax1,
    color='black',
    markerfacecolor='#122bff',
    markeredgecolor='#122bff',
    vlines_kwargs={"colors": "#33FF57"}
)
for item in ax1.collections:
    # change the color of the confidence interval 
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")

plt.title(r'Autocorrelograma Total Returns Yen Japonés $\Delta e$', fontsize=18)
plt.xlabel("Lag",fontsize=17)  
plt.ylabel("ACF",fontsize=17)
plt.xticks(range(0, len(acf_vals), 2), fontsize=15)    # Eje X

# Plot PACF
# plt.subplot(1, 2, 2)
# plt.stem(
#     range(len(pacf_vals)),
#     pacf_vals,
#     basefmt=" ",
#     use_line_collection=True,
#     linefmt="#33FF57",
#     markerfmt="o",  # Marker shape
# )
# plt.gca().get_children()[1].set_color("#122bff")  # Color markers
ax1 = plt.subplot(1, 2, 2)
plot_pacf(
    series,
    lags=20,
    ax=ax1,
    color='black',
    markerfacecolor='#122bff',
    markeredgecolor='#122bff',
    vlines_kwargs={"colors": "#33FF57"}
)
for item in ax1.collections:
    # change the color of the confidence interval 
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")

plt.title(r'Autocorrelograma Parcial Returns Yen Japonés $\Delta e$', fontsize=18)
plt.xlabel("Lag",fontsize=17)  
plt.ylabel("PACF",fontsize=17)
plt.xticks(range(0, len(acf_vals), 2), fontsize=15)    # Eje X

plt.tight_layout()
plt.savefig("acf_pacf_plot_spot_returns.svg", format='svg')
plt.show()


""" ~~~~~~~~~~~~ Estimación ~~~~~~~~~~~~ """


series = df_inflation_monthly['OBS_VALUE'].dropna()

modelo_ar2_inflation = AutoReg(series, lags=2)  # AR(2)
modelo_estimado = modelo_ar2_inflation.fit()

print(modelo_estimado.summary())

coeficientes_estimados = modelo_estimado.params
errores_estandar = modelo_estimado.bse
pvalues = modelo_estimado.pvalues
columna_1 = (r"$\phi_0$", r"$\phi_1$", r"$\phi_2$")

# Latex Table
coef_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lcccc}\n"
coef_table += "\\hline\n\\textbf{Variable} & \\textbf{coef.} & \\textbf{std. err.} & \\textbf{p-value} \\\\\n"
coef_table += "\\hline\n"
for var, coef, se, p in zip(columna_1, coeficientes_estimados, errores_estandar, pvalues):
    coef_table += f"{var} & {coef:.4f} & {se:.4f} & {p:.4f} \\\\\n"
coef_table += "\\hline\n\\end{tabular}\n\\caption{Resultados Estimación $AR(2)$ Inflación Japón.}\n\\end{table}\n"

tex_code = coef_table

with open("estimation_ar2_inflation.tex", "w") as f:
    f.write(tex_code)

print("TeX generado: 'estimation_ar2_inflation.tex'.")


# Coeficientes Lag Polinomio
ar_params = modelo_estimado.params.filter(like='L')

# Characteristic equation polynomial: 1 - phi_1*L - phi_2*L^2 = 0
ar_poly = np.r_[1, -ar_params.values]

# Compute the roots
roots = np.roots(ar_poly)
roots = (1 / roots)
# Roots consistentes con la documentación oficial:
# https://www.statsmodels.org/v0.12.2/generated/statsmodels.tsa.ar_model.ARResults.roots.html


# Latex Table
roots_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccc}\n"
roots_table += "\\hline\n\\textbf{Root} & \\textbf{Real} & \\textbf{Imaginary} & \\textbf{Modulus} \\\\\n"
roots_table += "\\hline\n"
for i, root in enumerate(roots, 1):
    modulus = abs(root)
    frequency = 0 if root.imag == 0 else np.angle(root) / (2 * np.pi)
    roots_table += f"AR.{i} & {root.real:.4f} & {root.imag:.4f} & {modulus:.4f} \\\\\n"
roots_table += "\\hline\n\\end{tabular}\n\\caption{Roots Modelo $AR(2)$ Inflación Japón.}\n\\end{table}\n"


with open("roots_ar2_inflation.tex", "w") as f:
    f.write(roots_table)

print("TeX generado: 'roots_ar2_inflation.tex'.")
print()
print()
print()


series_returns = df_exchange_rate_index_monthly['Percentage_Change'].dropna()

# ARMA(1,1) (ARIMA c/ d=0)
# modelo_arma_spot_returns = ARIMA(series_returns, order=(1, 0, 1))  # ARMA(1,1)
# modelo_arma_estimado = modelo_arma_spot_returns.fit()

modelo_ar1_return = AutoReg(series_returns, lags=1)  # AR(1)
modelo_ar1_estimado = modelo_ar1_return.fit()

# print(modelo_arma_estimado.summary())
print(modelo_ar1_estimado.summary())

# coeficientes_estimados = modelo_arma_estimado.params
# errores_estandar = modelo_arma_estimado.bse
# pvalues = modelo_arma_estimado.pvalues
# columna_1 = (r"$c$", r"$\phi_1$", r"$\theta_1$", r"$\sigma^2$")

coeficientes_estimados = modelo_ar1_estimado.params
errores_estandar = modelo_ar1_estimado.bse
pvalues = modelo_ar1_estimado.pvalues
columna_1 = (r"$\phi_0$", r"$\phi_1$")


# # Latex Table
# coef_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lcccc}\n"
# coef_table += "\\hline\n\\textbf{Variable} & \\textbf{coef.} & \\textbf{std. err.} & \\textbf{p-value} \\\\\n"
# coef_table += "\\hline\n"
# for var, coef, se, p in zip(columna_1, coeficientes_estimados, errores_estandar, pvalues):
#     coef_table += f"{var} & {coef:.4f} & {se:.4f} & {p:.4f} \\\\\n"
# coef_table += "\\hline\n\\end{tabular}\n\\caption{Resultados Estimación ARMA(1,1) Retornos Tipo de Cambio Yen.}\n\\end{table}\n"

# tex_code = coef_table

# with open("estimation_arma_results.tex", "w") as f:
#     f.write(tex_code)

# print("TeX generado: 'estimation_arma_results.tex'.")


# Latex Table
coef_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lcccc}\n"
coef_table += "\\hline\n\\textbf{Variable} & \\textbf{coef.} & \\textbf{std. err.} & \\textbf{p-value} \\\\\n"
coef_table += "\\hline\n"
for var, coef, se, p in zip(columna_1, coeficientes_estimados, errores_estandar, pvalues):
    coef_table += f"{var} & {coef:.4f} & {se:.4f} & {p:.4f} \\\\\n"
coef_table += "\\hline\n\\end{tabular}\n\\caption{Resultados Estimación AR(1) Retornos Tipo de Cambio Yen.}\n\\end{table}\n"

tex_code = coef_table

with open("estimation_ar1_results.tex", "w") as f:
    f.write(tex_code)

print("TeX generado: 'estimation_ar1_results.tex'.")


# Coeficientes Lag Polinomio
ar_params = modelo_ar1_estimado.params.filter(like='L')

# Characteristic equation polynomial: 1 - phi_1*L - phi_2*L^2 = 0
ar_poly = np.r_[1, -ar_params.values]

# Compute the roots
roots = np.roots(ar_poly)
roots = (1 / roots)
# Roots ... con la documentación oficial:
# https://www.statsmodels.org/v0.12.2/generated/statsmodels.tsa.ar_model.ARResults.roots.html


# Latex Table
roots_table = "\\begin{table}[ht]\n\\centering\n\\begin{tabular}{lccc}\n"
roots_table += "\\hline\n\\textbf{Root} & \\textbf{Real} & \\textbf{Imaginary} & \\textbf{Modulus} \\\\\n"
roots_table += "\\hline\n"
for i, root in enumerate(roots, 1):
    modulus = abs(root)
    frequency = 0 if root.imag == 0 else np.angle(root) / (2 * np.pi)
    roots_table += f"AR.{i} & {root.real:.4f} & {root.imag:.4f} & {modulus:.4f} \\\\\n"
roots_table += "\\hline\n\\end{tabular}\n\\caption{Roots Modelo $AR(1)$ Tipo de Cambio Japón.}\n\\end{table}\n"


with open("roots_ar1_returns.tex", "w") as f:
    f.write(roots_table)

print("TeX generado: 'roots_ar1_returns.tex'.")
print()
print()
print()


# Ljung-Box (Q) test results
# If the p-values are small (e.g., 𝑝 < 0.05), the residuals show significant autocorrelation, indicating the model may be misspecified.


# # Residuals ARMA
# residuals = modelo_arma_estimado.resid

# plt.figure(figsize=(12, 6))
# plt.plot(df_exchange_rate_index_monthly['TIME_PERIOD'][1:], residuals, label=r"Residuos", color="black")
# plt.axhline(0, color="black")
# plt.xlabel('Time')
# plt.title(r"Residuos $ARMA(1,1)$ Retornos Yen Japonés $\Delta e$")
# plt.legend()
# ax = plt.gca()  # Get current axis
# ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
# plt.xticks(rotation=45, ha='right')
# plt.tight_layout()
# plt.savefig("residuals_arma_1_1.svg", format='svg')
# plt.show()


# Residuals ARMA
residuals = modelo_ar1_estimado.resid

plt.figure(figsize=(12, 6))
# print(residuals.shape)
# print(df_exchange_rate_index_monthly['TIME_PERIOD'][1:].shape)
plt.plot(df_exchange_rate_index_monthly['TIME_PERIOD'][2:], residuals, label=r"Residuos", color="black")
plt.axhline(0, color="black")
plt.xlabel('Time')
plt.title(r"Residuos $AR(1)$ Retornos Yen Japonés $\Delta e$")
plt.legend()
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("residuals_ar_1.svg", format='svg')
plt.show()


# ACF and PACF plots
plt.subplots(figsize=(14, 6))

# Plot ACF
ax1 = plt.subplot(1, 2, 1)
plot_acf(
    residuals,
    lags=20,
    ax=ax1,
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
    vlines_kwargs={"colors": "black"},
)
for item in ax1.collections:
    # change the color of the confidence interval
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")
plt.title(r"Autocorrelograma Total Residuos $\Delta \hat{e}$", fontsize=18)
plt.xlabel("Lag", fontsize=17)
plt.ylabel("ACF", fontsize=17)
plt.xticks(range(0, 21, 2), fontsize=15)

# Plot PACF
ax2 = plt.subplot(1, 2, 2)
plot_pacf(
    residuals,
    lags=20,
    ax=ax2,
    color="black",
    markerfacecolor="black",
    markeredgecolor="black",
    vlines_kwargs={"colors": "black"},
)
for item in ax2.collections:
    # change the color of the confidence interval
    if type(item) == PolyCollection:
        item.set_facecolor("#c0c0c0")
plt.title(r"Autocorrelograma Parcial Residuos $\Delta \hat{e}$", fontsize=18)
plt.xlabel("Lag", fontsize=17)
plt.ylabel("PACF", fontsize=17)
plt.xticks(range(0, 21, 2), fontsize=15)

plt.tight_layout()
plt.savefig("acf_pacf_residuals.svg", format="svg")
plt.show()


# # Writing results:

# white_noise_model = ARIMA(series, order=(0, 0, 0)).fit()

# # Test de comparación contra ruido blanco ARMA(1,1)
# lrt_stat = 2 * (modelo_arma_estimado.llf - white_noise_model.llf)
# df = modelo_estimado.df_model - white_noise_model.df_model
# p_value = chi2.sf(lrt_stat, df)

# # If p < 0.05, the ARMA model is significantly better than white noise.
# print(f"LRT Statistic: {lrt_stat}, p-value: {p_value}")

# hypothesis_tex = r"""
# El diseño de las hipótesis del test está dado por:
# \newline

# \begin{center}
#     $H_0$: La serie sigue un modelo de ruido blanco (i.e., no hay estructura adicional). \\
#     $H_1$: La serie sigue un modelo ARMA(1,1) con una estructura significativa. \\
# \end{center}
# """

# filename_hipotesis = "hipotesis_test_ruido_blanco.tex"

# with open(filename_hipotesis, 'w', encoding='utf-8') as f_hypothesis:
#     f_hypothesis.write(hypothesis_tex)
# print(f"LaTeX generado: {filename_hipotesis}")


# results_filename = "tabla_lrt_results.tex"


# tex_content = r"""
# \begin{center}
#     \centering
#     \begin{tabular}{cc}
#         Likelihood Ratio Test (LRT) para ARMA(1,1) \\ \midrule
#         Estadístico LRT & """ + f"{lrt_stat:.6f} \\\\\n" + r"""
#         p-valor & """ + f"{p_value:.6f} \\\\\n" + r"""
#         \bottomrule
#     \end{tabular}
# \end{center}
# \vspace{10pt}
# """

# with open(results_filename, "w", encoding="utf-8") as f:
#     f.write(tex_content)


# print("Quiting:")
# print("(Breakpoint)")


# quit()


# Writing results:

white_noise_model = ARIMA(series, order=(0, 0, 0)).fit()

# Test de comparación contra ruido blanco ARMA(1,1)
lrt_stat = 2 * (modelo_ar1_estimado.llf - white_noise_model.llf)
df = modelo_ar1_estimado.df_model - white_noise_model.df_model
p_value = chi2.sf(lrt_stat, df)

# If p < 0.05, the AR model is significantly better than white noise.
print(f"LRT Statistic: {lrt_stat}, p-value: {p_value}")

hypothesis_tex = r"""
El diseño de las hipótesis del test está dado por:
\newline

\begin{center}
    $H_0$: La serie sigue un modelo de ruido blanco (i.e., no hay estructura adicional). \\
    $H_1$: La serie sigue un modelo ARMA(1,1) con una estructura significativa. \\
\end{center}
"""

filename_hipotesis = "hipotesis_test_ruido_blanco.tex"

with open(filename_hipotesis, 'w', encoding='utf-8') as f_hypothesis:
    f_hypothesis.write(hypothesis_tex)
print(f"LaTeX generado: {filename_hipotesis}")


results_filename = "tabla_lrt_results.tex"


tex_content = r"""
\begin{center}
    \centering
    \begin{tabular}{cc}
        Likelihood Ratio Test (LRT) para ARMA(1,1) \\ \midrule
        Estadístico LRT & """ + f"{lrt_stat:.6f} \\\\\n" + r"""
        p-valor & """ + f"{p_value:.6f} \\\\\n" + r"""
        \bottomrule
    \end{tabular}
\end{center}
\vspace{10pt}
"""

with open(results_filename, "w", encoding="utf-8") as f:
    f.write(tex_content)



# print("Quiting:")
# print("(Breakpoint)")


# quit()


"""
---------------------------------
       Models to simulate
---------------------------------

"""

print()

df_inflation_monthly['TIME_PERIOD'] = pd.to_datetime(df_inflation_monthly['TIME_PERIOD'])
index_dec_2018_Inf = df_inflation_monthly.loc[df_inflation_monthly['TIME_PERIOD'] == '2018-12'].index[0]
print("N tal que se cubre la muestra de Inflación hasta Diciembre-2018:", index_dec_2018_Inf)


df_exchange_rate_index_monthly['TIME_PERIOD'] = pd.to_datetime(df_exchange_rate_index_monthly['TIME_PERIOD'])
index_dec_2018_Exch = df_exchange_rate_index_monthly.loc[df_exchange_rate_index_monthly['TIME_PERIOD'] == '2018-12'].index[0]
print("N tal que se cubre la muestra de TC hasta Diciembre-2018:", index_dec_2018_Exch)


# Test:
# index_dec_2018_Inf =30
# index_dec_2018_Exch = 30

print()

df_inflation_monthly.to_csv('df_inflation_monthly_final.csv', index=False)

print("Csv generated: 'df_inflation_monthly_final.csv'")


df_inflation_monthly.to_csv('df_exchange_rate_index_monthly_final.csv', index=False)

print("Csv generated: 'df_exchange_rate_index_monthly_final.csv'")



# Coeficientes AR(2)

# AR polynomial: 1 - phi_1*L - phi_2*L^2
ar = np.array([modelo_estimado.params[0], -modelo_estimado.params[1], -modelo_estimado.params[2]])

# # Coeficientes ARMA(1,1)
# ar_arma = np.array([modelo_arma_estimado.params[0], -modelo_arma_estimado.params[1]])
# ma_arma = np.array([0, modelo_arma_estimado.params[2]])

# AR polynomial: 1 - phi_1*L - phi_2*L^2
ar_returns = np.array([modelo_ar1_estimado.params[0], -modelo_ar1_estimado.params[1]])


print("Simulating Inflation by Montecarlo...\n")

random.seed(10)

print(index_dec_2018_Inf)

monte_carlo_simulations_ar2 = monte_carlo_ar2(
    modelo_estimado.params[0],
    modelo_estimado.params[1],
    modelo_estimado.params[2],
    T=index_dec_2018_Inf,
    N=1000
)

# Probably it could be improvable

print("Simulating Inflation by Montecarlo Long Version...\n")

random.seed(10)

print(len(df_inflation_monthly))

monte_carlo_simulations_ar2_long_version = monte_carlo_ar2(
    modelo_estimado.params[0],
    modelo_estimado.params[1],
    modelo_estimado.params[2],
    T=len(df_inflation_monthly),
    N=1000
)


print("Ploting...\n")

plt.figure(figsize=(12, 6))
for sim in monte_carlo_simulations_ar2:
    plt.plot(df_inflation_monthly['TIME_PERIOD'][1:index_dec_2018_Inf],
             sim[1:], alpha=0.5, linewidth=0.8)

plt.title("Monte Carlo Simulations Inflación Japón.")
plt.xlabel("Time")
plt.ylabel(r"$\pi_t$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("inflation_monte_carlo.svg", format='svg')
plt.show()

# monte_carlo_simulations_arma_1_1 = monte_carlo_arma_1_1(
#     modelo_arma_estimado.params[0],
#     modelo_arma_estimado.params[1],
#     modelo_arma_estimado.params[2],
#     T=index_dec_2018_Exch,
#     N=1000
# )

print()

print("Simulating Exchange Rate Retunrs by Montecarlo...\n")

random.seed(10)

print(index_dec_2018_Exch)

monte_carlo_simulations_ar_1 = monte_carlo_ar_1(
    modelo_ar1_estimado.params[0],
    modelo_ar1_estimado.params[1],
    T=index_dec_2018_Exch,
    N=1000
)

# Probably it could be improvable

print("Simulating Exchange Rate Retunrs by Montecarlo Long Version...\n")

random.seed(10)

print(len(df_exchange_rate_index_monthly))

monte_carlo_simulations_ar1_long_version = monte_carlo_ar_1(
    modelo_ar1_estimado.params[0],
    modelo_ar1_estimado.params[1],
    T=len(df_exchange_rate_index_monthly),
    N=1000
)

print("Plotting...\n")

# plt.figure(figsize=(12, 6))
# for sim in monte_carlo_simulations_arma_1_1:
#     plt.plot(df_exchange_rate_index_monthly['TIME_PERIOD'][1:index_dec_2018_Exch],
#              sim[1:], alpha=0.5, linewidth=0.8)

plt.figure(figsize=(12, 6))
for sim in monte_carlo_simulations_ar_1:
    plt.plot(df_exchange_rate_index_monthly['TIME_PERIOD'][1:index_dec_2018_Exch],
             sim[1:], alpha=0.5, linewidth=0.8)

plt.title("Monte Carlo Simulations Retornos Yen.")
plt.xlabel("Time")
plt.ylabel(r"$\Delta e_t$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("returns_monte_carlo.svg", format='svg')
plt.show()


"""

_______________________________

    Estimations Inflation

_______________________________


"""

estimations_for_each_montecarlo = []

for simulation in monte_carlo_simulations_ar2:
    # print(simulation)
    modelo_ar2_inflation = AutoReg(simulation, lags=2)  # AR(2)
    modelo_estimado = modelo_ar2_inflation.fit()
    # print(modelo_estimado.summary())
    ar = np.array([modelo_estimado.params[0], modelo_estimado.params[1], modelo_estimado.params[2]])
    print(ar)
    estimations_for_each_montecarlo.append(ar)


estimations_array = np.array(estimations_for_each_montecarlo)

const_coeffs = estimations_array[:, 0]
phi1_coeffs = estimations_array[:, 1]
phi2_coeffs = estimations_array[:, 2]

plt.figure(figsize=(15, 10))

plt.suptitle(r"Distribuciones de $\hat{\phi}_0$, $\hat{\phi}_1$, y $\hat{\phi}_2$, Inflación Japón", fontsize=16)

plt.subplot(1, 3, 1)
sns.histplot(const_coeffs, bins=30, kde=True, color='#a5d7ff', stat="probability", linewidth=0)
# kde_kws={"color": "darkblue"}
plt.title("Distribución Estimación $\phi_0$")
plt.xlabel(r"$\hat{\phi}_0$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplot(1, 3, 2)
sns.histplot(phi1_coeffs, bins=30, kde=True, color='#a5d7ff', stat="probability", linewidth=0)
plt.title("Distribución Estimación $\phi_1$")
plt.xlabel(r"$\hat{\phi}_1$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplot(1, 3, 3)
sns.histplot(phi2_coeffs, bins=30, kde=True, color='#a5d7ff', stat="probability", linewidth=0)
plt.title("Distribución Estimación $\phi_2$")
plt.xlabel(r"$\hat{\phi}_2$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain') 

plt.subplots_adjust(hspace=0.3)

plt.savefig("monte_carlo_estimations_inflation.svg", format='svg')

plt.show()


"""

__________________________________

    Estimations Exch Rate Ret

__________________________________


"""


estimations_for_each_montecarlo = []

# for simulation in monte_carlo_simulations_arma_1_1:
#     # print(simulation)
#     # ARMA(1,1) (ARIMA c/ d=0)
#     modelo_arma_spot_returns = ARIMA(simulation, order=(1, 0, 1))  # ARMA(1,1)
#     modelo_arma_estimado = modelo_arma_spot_returns.fit()

#     # print(modelo_arma_estimado .summary())

#     # ar_arma = np.array([modelo_arma_estimado.params[0], -modelo_arma_estimado.params[1]])
#     # ma_arma = np.array([0, modelo_arma_estimado.params[2]])
#     arma = np.array([modelo_arma_estimado.params[0], modelo_arma_estimado.params[1], modelo_arma_estimado.params[2]])
#     print(arma)
#     estimations_for_each_montecarlo.append(arma)


for simulation in monte_carlo_simulations_ar_1:
    # print(simulation)
    # ARMA(1,1) (ARIMA c/ d=0)
    modelo_ar1_spot_returns = ARIMA(simulation, order=(1, 0, 0))  # AR(1)
    modelo_ar1_estimado = modelo_ar1_spot_returns .fit()

    arma = np.array([modelo_ar1_estimado.params[0], modelo_ar1_estimado.params[1], modelo_ar1_estimado.params[2]])
    print(arma)
    estimations_for_each_montecarlo.append(arma)


estimations_array = np.array(estimations_for_each_montecarlo)

const_coeffs = estimations_array[:, 0]
phi1_coeffs = estimations_array[:, 1]
theta2_coeffs = estimations_array[:, 2]

plt.figure(figsize=(15, 10))

# plt.suptitle(r"Distribuciones de $\hat{\phi}_0$, $\hat{\phi}_1$, y $\hat{\theta}_1$, Retornos Yen", fontsize=16)

# plt.subplot(1, 3, 1)
# sns.histplot(const_coeffs, bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
# plt.title("Distribución Estimación $\phi_0$")
# plt.xlabel(r"$\hat{\phi}_0$")
# plt.ylabel("Probability")
# plt.ticklabel_format(axis='y', style='plain')  

# plt.subplot(1, 3, 2)
# sns.histplot(phi1_coeffs, bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
# plt.title("Distribución Estimación $\phi_1$")
# plt.xlabel(r"$\hat{\phi}_1$")
# plt.ylabel("Probability")
# plt.ticklabel_format(axis='y', style='plain')

# plt.subplot(1, 3, 3)
# sns.histplot(phi2_coeffs, bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
# plt.title(r"Distribución Estimación $\theta_1$")
# plt.xlabel(r"$\hat{\phi}_2$")
# plt.ylabel("Probability")
# plt.ticklabel_format(axis='y', style='plain')  


plt.suptitle(r"Distribuciones de $\hat{\phi}_0$ y $\hat{\phi}_1$, Retornos Yen", fontsize=16)

plt.subplot(1, 2, 1)
sns.histplot(const_coeffs, bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
plt.title("Distribución Estimación $\phi_0$")
plt.xlabel(r"$\hat{\phi}_0$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')  

plt.subplot(1, 2, 2)
sns.histplot(phi1_coeffs, bins=30, kde=True, color='#d5ffa5', stat="probability", linewidth=0)
plt.title("Distribución Estimación $\phi_1$")
plt.xlabel(r"$\hat{\phi}_1$")
plt.ylabel("Probability")
plt.ticklabel_format(axis='y', style='plain')


plt.subplots_adjust(hspace=0.3)

plt.savefig("monte_carlo_estimations_spot_returns.svg", format='svg')

plt.show()


time_periods = df_inflation_monthly['TIME_PERIOD'][0:index_dec_2018_Inf].reset_index(drop=True)

simulations_ar2 = pd.DataFrame({'Time': time_periods})

for i in range(monte_carlo_simulations_ar2.shape[0]):
    simulations_ar2[f"sim_{i+1}"] = monte_carlo_simulations_ar2[i]

simulations_ar2.to_csv('simulations_ar2.csv', index=False)

print("Csv generated: 'simulations_ar2.csv'")


time_periods = df_exchange_rate_index_monthly['TIME_PERIOD'][0:index_dec_2018_Exch].reset_index(drop=True)

simulations_ar1 = pd.DataFrame({'Time': time_periods})

# simulations_arma11 = pd.DataFrame({'Time': time_periods})

# for i in range(monte_carlo_simulations_arma_1_1.shape[0]):
#     simulations_arma11[f"sim_{i+1}"] = monte_carlo_simulations_arma_1_1[i]

# simulations_arma11.to_csv('simulations_arma11.csv', index=False)

# print("Csv generate: 'simulations_arma11.csv'")


for i in range(monte_carlo_simulations_ar_1.shape[0]):
    simulations_ar1[f"sim_{i+1}"] = monte_carlo_simulations_ar_1[i]

simulations_ar1.to_csv('simulations_ar1.csv', index=False)

print("Csv generated: 'simulations_ar1.csv'")



time_periods = df_inflation_monthly['TIME_PERIOD'][0:len(df_inflation_monthly)].reset_index(drop=True)

simulations_ar2_long_version = pd.DataFrame({'Time': time_periods})

for i in range(monte_carlo_simulations_ar2_long_version.shape[0]):
    simulations_ar2_long_version[f"sim_{i+1}"] = monte_carlo_simulations_ar2_long_version[i]

simulations_ar2_long_version.to_csv('simulations_ar2_long_version.csv', index=False)

print("Csv generated: 'simulations_ar2_long_version.csv'")



time_periods = df_exchange_rate_index_monthly['TIME_PERIOD'][0:len(df_exchange_rate_index_monthly)].reset_index(drop=True)

simulations_ar1_long_version = pd.DataFrame({'Time': time_periods})

for i in range(monte_carlo_simulations_ar1_long_version.shape[0]):
    simulations_ar1_long_version[f"sim_{i+1}"] = monte_carlo_simulations_ar1_long_version[i]

simulations_ar1_long_version.to_csv('simulations_ar1_long_version.csv', index=False)

print("Csv generated: 'simulations_ar1_long_version.csv'")




"""

Pregunta 2.d

_____________________________________________________________________________________________________

# Este código a la actualidad no funciona; fue reemplazado por
# `Courses/Econometric_Theory_III_EAE4103-2024-2/homework_assignments/T1/pregunta_2.m`

# Además se diseñó en un principio con ARMAs, antes de volver a un setup 100% AR para ambas series

_____________________________________________________________________________________________________

"""


# "Tic"
# start_time_ar2 = datetime.now()

# length_aic_ar2 = []
# length_bic_ar2 = []

# for simulation in monte_carlo_simulations_ar2:
#     time_series = simulation

#     best_aic = 26
#     best_bic = 26
#     best_order_aic = (0, 0, 0)
#     best_order_bic = (0, 0, 0)


#     # Grid search over p and q
#     for p in range(25):
#         for q in range(25):
#             try:
#                 model = ARIMA(time_series, order=(p, 0, q))
#                 model_fitted = model.fit()
#                 aic = model_fitted.aic
#                 bic = model_fitted.bic
                
#                 if aic < best_aic:
#                     best_aic = aic
#                     best_order_aic = (p, 0, q)
                
#                 if bic < best_bic:
#                     best_bic = bic
#                     best_order_bic = (p, 0, q)

#             except:
#                 continue

#     print(f"Lags según AIC (AR2): {best_order_aic}")
#     print(f"Lags según BIC (AR2): {best_order_bic}")

#     length_aic_ar2.append(best_order_aic)
#     length_bic_ar2.append(best_order_bic)

# end_time_ar2 = datetime.now()


# print('Duration execution Loops AIC/BIC for AR2: {}'.format(end_time_ar2 - start_time_ar2))

# start_time_arma11 = datetime.now()

# length_aic_arma11 = []
# length_bic_arma11 = []


# for simulation in monte_carlo_simulations_arma_1_1:
#     time_series = simulation

#     best_aic = 26
#     best_bic = 26
#     best_order_aic = (0, 0, 0)
#     best_order_bic = (0, 0, 0)


#     # Grid search over p and q
#     for p in range(25):
#         for q in range(25):
#             try:
#                 model = ARIMA(time_series, order=(p, 0, q))
#                 model_fitted = model.fit()
#                 aic = model_fitted.aic
#                 bic = model_fitted.bic
                
#                 if aic < best_aic:
#                     best_aic = aic
#                     best_order_aic = (p, 0, q)
                
#                 if bic < best_bic:
#                     best_bic = bic
#                     best_order_bic = (p, 0, q)

#             except:
#                 continue

#     print(f"Lags según AIC (ARMA1): {best_order_aic}")
#     print(f"Lags según BIC (ARMA1): {best_order_bic}")

#     length_aic_arma11.append(best_order_aic)
#     length_bic_arma11.append(best_order_bic)

# end_time_arma11 = datetime.now()

# print('Duration execution Loops AIC/BIC for ARMA 1,1: {}'.format(end_time_arma11 - start_time_arma11))

# data_to_save = {
#     'length_aic_ar2': length_aic_ar2,
#     'length_bic_ar2': length_bic_ar2,
#     'end_time_ar2': end_time_ar2,
#     'execution_time_aic_bic_ar2': end_time_ar2 - start_time_ar2,
#     'length_aic_arma11': length_aic_arma11,
#     'length_bic_arma11': length_bic_arma11,
#     'start_time_arma11': start_time_arma11,
#     'end_time_arma11': end_time_arma11,
#     'execution_time_aic_bic_arma11': end_time_arma11 - start_time_arma11
# }

# with open('simulation_results.pkl', 'wb') as f:
#     pickle.dump(data_to_save, f)

# joblib.dump(data_to_save, 'simulation_results.pkl')

