import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def ajustar(temperaturas, d, w):


    A = np.column_stack([
        np.ones(n),
        np.cos(w*d),
        np.sin(w*d)
    ])

    coef = np.linalg.solve(A.T @ A, A.T @ temperaturas)

    return coef

#nome da tabela com os dados simplificados no meu computador
df = pd.read_csv(r"dados_simplificados.csv")

Tmin = df["Tmin"].to_numpy()
Tmax = df["Tmax"].to_numpy()

n = len(Tmin)
d = np.arange(n)
w = 2*np.pi/365

theta = ajustar(Tmin, d, w)
phi = ajustar(Tmax, d, w)

print("Coeficientes Tmin:")
print(theta)

print("\nCoeficientes Tmax:")
print(phi)

#Grafico

Tmin_ajustada = (
    theta[0]
    + theta[1]*np.cos(w*d)
    + theta[2]*np.sin(w*d)
)

Tmax_ajustada = (
    phi[0]
    + phi[1]*np.cos(w*d)
    + phi[2]*np.sin(w*d)
)

#grafico das temperaturas minimas
plt.figure(figsize=(10, 6))

plt.scatter(
    d,
    Tmin,
    s=10,
    label="Dados experimentais"
)

plt.plot(
    d,
    Tmin_ajustada,
    linewidth=2,
    label="Ajuste MMQ"
)

plt.xlabel("Dia do ano")
plt.ylabel("Temperatura minima (°C)")
plt.title("Ajuste periódico das temperaturas minimas")
plt.legend()
plt.grid()

plt.show()

#grafico das temperaturas maximas
plt.figure(figsize=(10, 6))

plt.scatter(
    d,
    Tmax,
    s=10,
    label="Dados experimentais"
)

plt.plot(
    d,
    Tmax_ajustada,
    linewidth=2,
    label="Ajuste MMQ"
)

show_temps = False

if show_temps:
    plt.xlabel("Dia do ano")
    plt.ylabel("Temperatura maxima (°C)")
    plt.title("Ajuste periódico das temperaturas maximas")
    plt.legend()
    plt.grid()
    plt.show()



df = pd.DataFrame({
    "hora": np.arange(0.5, 24, 1.0),
    "hora_norm": np.arange(0.5, 24, 1.0) / 24,
    "T_verao": [20.5, 20.1, 19.9, 19.8, 19.6, 19.0, 19.0, 19.9, 21.8, 23.7, 25.0, 26.1, 27.0, 27.7, 27.8, 27.7, 27.2, 26.8, 25.8, 23.9, 22.2, 21.9, 21.7, 20.8],
    "T_inverno": [15.5, 14.8, 13.2, 14.0, 13.8, 13.4, 13.2, 13.2, 14.2, 17.0, 19.9, 20.2, 23.9, 24.9, 25.7, 25.9, 25.9, 25.2, 23.8, 21.2, 19.4, 17.9, 16.8, 16.1]
})
