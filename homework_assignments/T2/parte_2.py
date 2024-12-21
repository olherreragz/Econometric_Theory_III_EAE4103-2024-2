import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import adfuller


file_path = 'data/database.xlsx'
df = pd.read_excel(file_path)

# first rows
print(df.head())


plt.figure(figsize=(10, 6))
plt.plot(
    df['date'],
    df['copper'],
    label='Precio delCobre',
    color='#ffc49d'
)
plt.xlabel('Time')
plt.ylabel(r'Precio')
plt.title(r'Evolución del Precio del Cobre Internacional')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("copper_price.svg", format='svg')
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

Serie = "Precio del Cobre Internacional"

print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df['copper'].dropna(), autolag='AIC')
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

results_filename="tabla_adf_results_copper.tex"

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

df['copper_returns'] = df['copper'].pct_change() * 100


print("Copper Returns:")
print(df[['date', 'copper']])


plt.figure(figsize=(10, 6))
plt.plot(
    df['date'],
    df['copper_returns'],
    label='Returns Copper', color='#ffc49d'
)
plt.xlabel('Time')
plt.ylabel(r'$\Delta \%$')
plt.title(r'Evolución de Retornos del Precio del Cobre')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("copper_returns.svg", format='svg')
plt.show()


# (A Little hardcoded):

Serie = "Retornos del Precio del Cobre"

print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df['copper_returns'].dropna(), autolag='AIC')
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

results_filename="tabla_adf_results_copper_returns.tex"

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


plt.figure(figsize=(10, 6))
plt.plot(
    df['date'],
    df['exchange_rate'],
    label='Tipo de Cambio USD,CLP', color='black'
)
plt.xlabel('Time')
plt.ylabel('USD,CLP')
plt.title(r"Evolución del Tipo de Cambio de Chile ($USD,CLP$)")
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("chile_spot.svg", format='svg')
plt.show()


# (A Little hardcoded):

Serie = "Tipo de Cambio USD,CLP"

print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df['exchange_rate'].dropna(), autolag='AIC')
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



df['spot_returns'] = df['exchange_rate'].pct_change() * 100

print("Retornos USD,CLP:")
print(df[['date', 'spot_returns']])


plt.figure(figsize=(10, 6))
plt.plot(
    df['date'],
    df['spot_returns'],
    label='Retornos USD,CLP', color='black'
)
plt.xlabel('Time')
plt.ylabel(r'$\Delta e$')
plt.title(r'Retornos $\Delta e$ ($USD,CLP$)')
ax = plt.gca()  # Get current axis
ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both'))  # Automatically manage the x-ticks
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("chilean_depreciation.svg", format='svg')
plt.show()



# (A Little hardcoded):

Serie = "Retornos Tipo de Cambio USD,CLP"

print(f"Augmented Dickey-Fuller test for \"{Serie}\":")
result = adfuller(df['spot_returns'].dropna(), autolag='AIC')
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
    

df['date'] = pd.to_datetime(df['date'], format='%b.%Y')


output_path = 'data/database_final.xlsx'

df.to_excel(output_path, index=False)

print(f"DataFrame saved successfully to database_final.xlsx")


