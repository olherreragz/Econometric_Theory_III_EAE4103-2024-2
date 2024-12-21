import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess


def acs_to_tex(acf_vals, pacf_vals, filename="tabla_acf_pacf_ar2.tex"):
    with open(filename, 'w', encoding='utf-8') as f:
        # f.write(r"\begin{table}[h!]" + "\n")
        # f.write(r"\centering" + "\n")
        # f.write(r"\begin{tabular}{|c|c|c|}" + "\n")
        # f.write(r"\hline" + "\n")
        # f.write(r"Lag & ACF $\gamma_j$ & PACF $\phi_{j,j}$\\" + "\n")
        # f.write(r"\hline" + "\n")
        
        # for i in range(len(acf_vals)):
        #     f.write(f"{i} & {acf_vals[i]:.4f} & {pacf_vals[i]:.4f} \\\\" + "\n")
        
        # f.write(r"\hline" + "\n")
        # f.write(r"\end{tabular}" + "\n")
        # f.write(r"\caption{ACF y PACF teóricos para un AR(2) con $\phi_1 = 0.6$ y $\phi_1 = 0.6$}" + "\n")
        # f.write(r"\end{table}" + "\n")

        # Table start
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{ACF y PACF teóricos para un AR(2) con $\phi_1 = 0.6$ y $\phi_2 = 0.2$}" + "\n")
        f.write(r"\label{tab:acf_pacf_ar2}" + "\n")
        f.write(r"\begin{tabular}{ccc}" + "\n")
        f.write(r"\toprule" + "\n")        
        f.write(r"Lag & ACF $\gamma_j$ & PACF $\phi_{j,j}$ \\" + "\n")
        f.write(r"\midrule" + "\n")
        
        for i in range(len(acf_vals)):
            f.write(f"{i} & {acf_vals[i]:.4f} & {pacf_vals[i]:.4f} \\\\" + "\n")
        
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

# Coeficientes AR(2)
phi_1 = 0.6
phi_2 = 0.2

# AR polynomial: 1 - phi_1*L - phi_2*L^2
ar = np.array([1, -phi_1, -phi_2])
ma = np.array([1])  # No rezagos MAs

ar2_process = ArmaProcess(ar, ma)

# autocorrelogramas totales y autocorrelogramas parciales teóricos

acf_vals = ar2_process.acf(20)                      # Número de lags a computar como argumento
pacf_vals = ar2_process.pacf(20)

print("\nValores ACF:")
print(acf_vals)

print("\nValoresl PACF:")
print(pacf_vals)

acs_to_tex(acf_vals, pacf_vals)

# Plot ACF
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.stem(
    range(len(acf_vals)),
    acf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#38f70d",
    markerfmt="o",       # Marker shape
)
plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
plt.title("Autocorrelograma Total Teórico")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.xticks(range(0, len(acf_vals), 2))  # Eje X

# Plot PACF
plt.subplot(1, 2, 2)
plt.stem(
    range(len(pacf_vals)),
    pacf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#38f70d",
    markerfmt="o",       # Marker shape
)
plt.gca().get_children()[1].set_color("#122bff")  # Color markers
plt.title("Autocorrelograma Parcial Teórico")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.xticks(range(0, len(acf_vals), 2))  # Eje X

plt.tight_layout()
# plt.show()

plt.savefig("acf_pacf_plot.svg", format='svg')

