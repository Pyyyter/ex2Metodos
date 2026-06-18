# Exercício 2 - Métodos Numéricos
# Definições
TEMPERATURA_DE_CONFORTO = 24
fac = 8800
nFac = 1/3
show_temps = True
# dados do ambiente
comprimento = 8
largura = 6
altura = 3

heff = 8.0                  # W/(m² K)
Fger = 1800.0               # W

# área total de troca com o ambiente externo
Atot = 2 * (comprimento * largura + comprimento * altura + largura * altura)

# Questões 1 e 2
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


if show_temps:
    plt.xlabel("Dia do ano")
    plt.ylabel("Temperatura maxima (°C)")
    plt.title("Ajuste periódico das temperaturas maximas")
    plt.legend()
    plt.grid()
    plt.show()
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

    if show_temps:
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










# Questão 3
from scipy.interpolate import CubicSpline

horas = np.arange(0.5, 24, 1.0)

temperaturas_verao = np.array([
    20.5, 20.1, 19.9, 19.8,
    19.6, 19.0, 19.0, 19.9,
    21.8, 23.7, 25.0, 26.1,
    27.0, 27.7, 27.8, 27.7,
    27.2, 26.8, 25.8, 23.9,
    22.2, 21.9, 21.7, 20.8
])

temperaturas_inverno = np.array([
    15.5, 14.8, 13.2, 14.0,
    13.8, 13.4, 13.2, 13.2,
    14.2, 17.0, 19.9, 20.2,
    23.9, 24.9, 25.7, 25.9,
    25.9, 25.2, 23.8, 21.2,
    19.4, 17.9, 16.8, 16.1
])

dados_diarios = pd.DataFrame({
    "hora": horas,
    "hora_normalizada": horas / 24,
    "verao": temperaturas_verao,
    "inverno": temperaturas_inverno
})

min_verao = dados_diarios["verao"].min()
max_verao = dados_diarios["verao"].max()

min_inverno = dados_diarios["inverno"].min()
max_inverno = dados_diarios["inverno"].max()

dados_diarios["verao_normalizado"] = (
    (dados_diarios["verao"] - min_verao) / (max_verao - min_verao)
)

dados_diarios["inverno_normalizado"] = (
    (dados_diarios["inverno"] - min_inverno) / (max_inverno - min_inverno)
)

x_horas = dados_diarios["hora_normalizada"].to_numpy()

x_periodico = np.append(x_horas, 1.0)
verao_periodico = np.append(
    dados_diarios["verao_normalizado"].to_numpy(),
    dados_diarios["verao_normalizado"].iloc[0]
)
inverno_periodico = np.append(
    dados_diarios["inverno_normalizado"].to_numpy(),
    dados_diarios["inverno_normalizado"].iloc[0]
)

spline_verao = CubicSpline(x_periodico, verao_periodico, bc_type="periodic")
spline_inverno = CubicSpline(x_periodico, inverno_periodico, bc_type="periodic")


if show_temps:
    x_plot = np.linspace(0, 1, 600)
    horas_plot = 24 * x_plot

    plt.figure(figsize=(11, 6))

    plt.scatter(
        dados_diarios["hora"],
        dados_diarios["verao_normalizado"],
        color="#d95f02",
        s=28,
        label="Dados normalizados - verão"
    )
    plt.plot(
        horas_plot,
        spline_verao(x_plot),
        color="#d95f02",
        linewidth=2.2,
        label="Spline cúbica periódica - verão"
    )

    plt.scatter(
        dados_diarios["hora"],
        dados_diarios["inverno_normalizado"],
        color="#1b9e77",
        s=28,
        label="Dados normalizados - inverno"
    )
    plt.plot(
        horas_plot,
        spline_inverno(x_plot),
        color="#1b9e77",
        linewidth=2.2,
        label="Spline cúbica periódica - inverno"
    )
    plt.xlabel("Hora do dia")
    plt.ylabel("Temperatura normalizada")
    plt.title("Representação gráfica das splines cúbicas periódicas")
    plt.xlim(0, 24)
    plt.ylim(-0.05, 1.05)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


#region Questão 4


def mascara_de_dia_util(dia):
    dia_da_semana = int(np.floor(dia)) % 7
    return 1 if dia_da_semana <= 4 else 0


def alpha(dia):
    dia = np.asarray(dia, dtype=float)
    resultado = np.empty_like(dia)

    trecho_1 = (dia >= 0) & (dia <= 171)
    trecho_2 = (dia > 171) & (dia <= 355)
    trecho_3 = (dia > 355) & (dia <= 365)

    resultado[trecho_1] = (19 / 3420) * dia[trecho_1] + 1 / 20
    resultado[trecho_2] = (-1 / 184) * dia[trecho_2] + 355 / 184
    resultado[trecho_3] = (1 / 200) * dia[trecho_3] - 355 / 200

    return resultado


def Tmin_modelo(dia):
    return theta[0] + theta[1] * np.cos(w * dia) + theta[2] * np.sin(w * dia)


def Tmax_modelo(dia):
    return phi[0] + phi[1] * np.cos(w * dia) + phi[2] * np.sin(w * dia)


def Te(dia, hora):
    hora_normalizada = np.asarray(hora, dtype=float) / 24.0

    perfil_verao = spline_verao(hora_normalizada)
    perfil_inverno = spline_inverno(hora_normalizada)

    temperatura_minima = Tmin_modelo(dia)
    temperatura_maxima = Tmax_modelo(dia)
    peso_inverno = alpha(dia)

    perfil_diario = (1 - peso_inverno) * perfil_verao + peso_inverno * perfil_inverno
    return perfil_diario * (temperatura_maxima - temperatura_minima) + temperatura_minima


def fger(dia, hora):
    hora = np.asarray(hora, dtype=float)
    horario_comercial = (hora >= 8) & (hora <= 18)
    dia_util = mascara_de_dia_util(dia) == 1
    ativo = horario_comercial & dia_util
    return np.where(ativo, Fger, 0.0)


def Ti(dia, hora):
    return Te(dia, hora) + fger(dia, hora) / (heff * Atot)


def p(dia, hora):
    hora = np.asarray(hora, dtype=float)
    ti = Ti(dia, hora)

    horario_comercial = (hora >= 8) & (hora <= 18)
    dia_util = mascara_de_dia_util(dia) == 1
    ligado = horario_comercial & dia_util & (ti > TEMPERATURA_DE_CONFORTO)

    return np.where(ligado, nFac * fac, 0.0)


#region Gráficos da questão 4

if show_temps:
    horas_plot = np.linspace(0, 24, 600)
    dias_escolhidos = [0, 80, 171, 260]

    # gráfico da temperatura externa
    plt.figure(figsize=(11, 6))
    for dia in dias_escolhidos:
        plt.plot(horas_plot, Te(dia, horas_plot), linewidth=2, label=f"Te - dia {dia}")

    plt.axhline(
        TEMPERATURA_DE_CONFORTO,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Temperatura de conforto"
    )

    plt.xlabel("Hora do dia")
    plt.ylabel("Temperatura externa estimada (°C)")
    plt.title("Questão 4 - Temperatura externa ao longo do dia")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # gráfico da temperatura interna
    plt.figure(figsize=(11, 6))
    for dia in dias_escolhidos:
        plt.plot(horas_plot, Ti(dia, horas_plot), linewidth=2, label=f"Ti - dia {dia}")

    plt.axhline(
        TEMPERATURA_DE_CONFORTO,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Temperatura de conforto"
    )

    plt.xlabel("Hora do dia")
    plt.ylabel("Temperatura interna estimada (°C)")
    plt.title("Questão 4 - Temperatura interna ao longo do dia")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # gráfico da potência do ar-condicionado
    plt.figure(figsize=(11, 6))
    for dia in dias_escolhidos:
        plt.plot(horas_plot, p(dia, horas_plot), linewidth=2, label=f"p - dia {dia}")

    plt.xlabel("Hora do dia")
    plt.ylabel("Potência elétrica do ar-condicionado (W)")
    plt.title("Questão 4 - Potência de uso do ar-condicionado")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#endregion
#endregion  








#region questão 5

#potencia em Wh usada pelo ar em 1 ano
P_total = 0.0

#o periodo de 1 dia e de meia hora
delta_d = 1
delta_h = 0.5

horas = np.arange(0, 24, delta_h)

for dia in range(365):
    p_dia = 0
    for hora in horas:
        p_dia += p(dia, hora) * delta_h
    P_total += p_dia * delta_d

print(f"\nO ar usou {P_total:.0f}W no ano")

#endregion

# Questão 6

CUSTO_KWH = 1.1

def gauss_legendre_2d(f, dia_ini, dia_fim, hora_ini, hora_fim, nd, nh):
    xd, wd = np.polynomial.legendre.leggauss(nd)
    xh, wh = np.polynomial.legendre.leggauss(nh)

    dias = 0.5 * (dia_fim - dia_ini) * xd + 0.5 * (dia_ini + dia_fim)
    horas = 0.5 * (hora_fim - hora_ini) * xh + 0.5 * (hora_ini + hora_fim)

    soma = 0.0
    for i in range(nd):
        for j in range(nh):
            soma += wd[i] * wh[j] * f(dias[i], horas[j])

    return 0.25 * (dia_fim - dia_ini) * (hora_fim - hora_ini) * soma

def potencia_total_gauss_blocos(n):
    total = 0.0
    for dia in range(365):
        if dia % 7 <= 4:   # segunda a sexta
            total += gauss_legendre_2d(p, dia, dia + 1, 8.0, 18.0, n, n)
    return total

ns = [4,5,6,7,8,9,10,11,12]
resultados = []

for n in ns:
    valor = potencia_total_gauss_blocos(n)
    resultados.append(valor)
    print(f"n = {n:2d} -> P_total = {valor:.2f} Wh")

valor_ref = potencia_total_gauss_blocos(32)

erros = []
for valor in resultados:
    erro_rel = abs((valor - valor_ref) / valor_ref) * 100
    erros.append(erro_rel)

n_ideal = None
for i, n in enumerate(ns):
    if erros[i] < 0.1:
        n_ideal = n
        break

if n_ideal is None:
    n_ideal = ns[-1]

P_total_gauss = potencia_total_gauss_blocos(n_ideal)
custo_anual = (P_total_gauss / 1000.0) * CUSTO_KWH

print("\nConvergência Gauss-Legendre 2D por blocos:")
for i, n in enumerate(ns):
    print(
        f"n = {n:2d} -> P_total = {resultados[i]:.2f} Wh | "
        f"erro relativo = {erros[i]:.6f}%"
    )

print(f"\nValor de referência: {valor_ref:.2f} Wh")
print(f"Número de pontos escolhido: n = {n_ideal}")
print(f"Potência total anual (Gauss-Legendre): {P_total_gauss:.2f} Wh")
print(f"Consumo anual: {P_total_gauss / 1000.0:.2f} kWh")
print(f"Custo anual estimado: R$ {custo_anual:.2f}")
# gráfico da potência total anual
plt.figure(figsize=(10, 5))
plt.plot(ns, resultados, marker="o", linewidth=2, color="#1f77b4")
plt.xlabel("Número de pontos por dimensão")
plt.ylabel("Potência total anual (Wh)")
plt.title("Convergência da quadratura de Gauss-Legendre 2D")
plt.xticks(ns)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# gráfico do erro relativo
plt.figure(figsize=(10, 5))
plt.plot(ns, erros, marker="o", linewidth=2, color="darkred")
plt.axhline(0.1, color="black", linestyle="--", linewidth=1.5, label="Tolerância = 0,1%")
plt.xlabel("Número de pontos por dimensão")
plt.ylabel("Erro relativo (%)")
plt.title("Erro relativo da quadratura de Gauss-Legendre 2D")
plt.xticks(ns)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()