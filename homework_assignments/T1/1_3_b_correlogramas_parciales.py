import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess


# Coeficientes AR(2)
phi_1 = 0
phi_2 = 0
phi_3 = 0
phi_4 = 0.5

# AR polynomial: 1 - phi_1*L - phi_2*L^2
ar = np.array([1, -phi_1, -phi_2, -phi_3, -phi_4])
ma = np.array([1])  # No rezagos MAs

ar4_process = ArmaProcess(ar, ma)

# Coeficientes MA(4)
theta_1 = 0
theta_2 = 0
theta_3 = 0
theta_4 = 0.5

# MA polynomial
ar = np.array([1])  # No AR
ma = np.array([1, theta_1, theta_2, theta_3, theta_4])

ma4_process = ArmaProcess(ar, ma)

# autocorrelogramas totales y autocorrelogramas parciales teóricos

ar4_acf_vals = ar4_process.acf(20)                      # Número de lags a computar como argumento
ar4_pacf_vals = ar4_process.pacf(20)

ma4_acf_vals = ma4_process.acf(20)                      # Número de lags a computar como argumento
ma4_pacf_vals = ma4_process.pacf(20)

plt.figure(figsize=(12, 6))

# Plot ACF MA(4)
plt.subplot(2, 2, 1)
plt.stem(
    range(len(ma4_acf_vals)),
    ma4_acf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#38f70d",  # MA(4) ACF color
    markerfmt="o",
)
plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
plt.title("Autocorrelograma Total Teórico MA(4)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.xticks(range(0, len(ma4_acf_vals), 2))  # Eje X

# Plot PACF MA(4)
plt.subplot(2, 2, 2)
plt.stem(
    range(len(ma4_pacf_vals)),
    ma4_pacf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#38f70d",  # MA(4) PACF color
    markerfmt="o",
)
plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
plt.title("Autocorrelograma Parcial Teórico MA(4)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.xticks(range(0, len(ma4_pacf_vals), 2))  # Eje X

# Plot ACF AR(4)
plt.subplot(2, 2, 3)
plt.stem(
    range(len(ar4_acf_vals)),
    ar4_acf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#ff1e28",  # AR(4) ACF color
    markerfmt="o",
)
plt.gca().get_children()[1].set_color("#122bff")  # Color Markers
plt.title("Autocorrelograma Total Teórico AR(4)")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.xticks(range(0, len(ar4_acf_vals), 2))  # Eje X

# Plot PACF AR(4)
plt.subplot(2, 2, 4)
plt.stem(
    range(len(ar4_pacf_vals)),
    ar4_pacf_vals,
    basefmt=" ",
    use_line_collection=True,
    linefmt="#ff1e28",  # AR(4) PACF color
    markerfmt="o",
)
plt.gca().get_children()[1].set_color("#122bff")  # Color markers
plt.title("Autocorrelograma Parcial Teórico AR(4)")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.xticks(range(0, len(ar4_pacf_vals), 2))  # Eje X

plt.tight_layout()
# plt.show()

plt.savefig("acf_pacf_ar_y_ma_plot.svg", format='svg')  # Save as SVG

