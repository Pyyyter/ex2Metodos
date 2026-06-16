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
df = pd.read_csv(r"C:\Users\nando\Downloads\dados_simplificados.csv")

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

plt.xlabel("Dia do ano")
plt.ylabel("Temperatura maxima (°C)")
plt.title("Ajuste periódico das temperaturas maximas")
plt.legend()
plt.grid()

plt.show()