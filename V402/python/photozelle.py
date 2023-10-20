import pandas as pd
from scipy import stats, optimize
from monke import functions, constants
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = [6.5, 5.5]
# plt.style.use("science")


def freq(wavelength: float):
    return constants.c / wavelength * 1e9


# maximaler Bereich der in den plots gezeigt wird
u_max = {
    "365": 3000,
    "405": 3000,
    "436": 3000,
    "546": 3000,
    "578": 3000,
}

# maximale Bereich, der zur linearen Regression genutzt wird
u_max_linear = {
    "365": 2000,
    "405": 1250,
    "436": 1100,
    "546": 350,
    "578": 300,
}

U0_data: dict[list] = {}
h = []


# Geraden-Fit
def linear(x, m, b):
    return m * x + b


"""Bestimme für alle Kennlinien die Grenzspannung und füge diese in das dict[list] 
U0 zur entsprechenden Wellenlänge hinzu"""
with pd.ExcelFile("../Daten/photozelle_kennlinie.xlsx") as file:
    for sheet in file.sheet_names:
        wavelength = sheet[:3]

        # Rohdaten
        data_raw: np.ndarray = np.array(
            pd.read_excel(file, sheet))  # 1: I, 2: I_err, 3: U
        data_raw = data_raw[data_raw[:, 2] < u_max[wavelength]]

        # Werte außerhalb des quadratischen Bereichs filtern
        data = pd.DataFrame(data_raw)

        # Berechne die Wurzel von I sowie den dazugehörigen Fehler und ziehe den Anodenstrom ab
        data[0] = data[0].apply(lambda x: np.sqrt(x - data[0].min()))
        data = data.head(-1)
        data[1] = data[1] / data[0]

        # Filtert alle Werte raus, wo die kennlinie einen linearen Anstieg hat
        ind = data[2] < u_max_linear[sheet[:3]]

        # Berechne die Ausgleichsgerade und speicher die grenzspannung in U0 ab
        result = stats.linregress(data[2][ind], data[0][ind])
        popt, pcov = optimize.curve_fit(
            linear, data[2][ind], data[0][ind])

        intercept, intercept_err = popt[1], pcov[1, 1]**0.5
        slope, slope_err = popt[0], pcov[0, 0]**0.5
        u0: float = - popt[1] / popt[0]
        u0_err: float = np.sqrt((intercept_err/slope) **
                                2 + (intercept*slope_err/slope**2)**2)
        if U0_data.get(wavelength):
            U0_data[wavelength].append((u0, u0_err))
        else:
            U0_data[wavelength] = [(u0, u0_err)]

        # Erstelle kennlinien plots
        fig, ax = plt.subplots()
        ax.errorbar(data[2], data[0], yerr=data[1],
                    marker="o", ms=5, linestyle="")
        xlim = ax.get_xlim()

        x = np.linspace(xlim[0], xlim[1], 4)

        ax.plot(x, intercept + slope * x,
                label=f"U0 = {functions.error_round(u0, u0_err, 'parenthesis')} [mV]")

        ax.set_xlim(xlim)
        ax.set_ylim((-0.5, ax.get_ylim()[1]))

        ax.set_xlabel(r"$-U$ [mV]")
        ax.set_ylabel(r"$\sqrt{I-I_0}$ [pA$^{1/2}$]")
        ax.legend()
        plt.savefig(f"../figs/photozelle_kennline_{sheet}.png")

U0 = []
for i in U0_data:
    data = [[u[0] for u in U0_data[i]], [u[1] for u in U0_data[i]]]
    u = np.array(data[0]).mean()
    u_err = np.array(data[1]).mean() + np.array(data[0]).std()
    U0.append([freq(float(i)), u, u_err])

U0: pd.DataFrame = pd.DataFrame(U0)
for i in range(1, 3):
    U0[i] = U0[i].map(lambda x: x / 1000) # Umrechnung in V
popt, pcov = optimize.curve_fit(linear, U0[0], U0[1], sigma=U0[2])
slope = (popt[0], pcov[0, 0]**0.5)
intercept = (popt[1], pcov[1, 1]**0.5)

planck_constant = (slope[0] * constants.q, slope[1] * constants.q)
work_function = (-intercept[0]*constants.q, intercept[1]*constants.q)

plt.subplots()
plt.errorbar(U0[0], U0[1], yerr=U0[2], marker="o", ms=5, linestyle="")
plt.plot(U0[0], U0[0]*slope[0] + intercept[0], 
         label=f"h = {functions.error_round(planck_constant[0], planck_constant[1], 'scientific')[0]} Js")
plt.legend()
plt.ylabel("$U_0$ [V]")
plt.xlabel(r"$\nu$ [Hz]")
plt.savefig("../figs/photozelle_wirkungsquantum.png", dpi=200)

print(f"h = {functions.error_round(planck_constant[0], planck_constant[1], 'scientific')[0]} Js")
print(f"W = {functions.error_round(work_function[0], work_function[1], 'scientific')[0]} J")