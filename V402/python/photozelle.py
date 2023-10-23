import pandas as pd
from scipy import odr
from monke import functions, constants
from monke.latex import *
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir(os.path.dirname(__file__))
plt.rcParams["figure.figsize"] = [6.5, 5.5]
# plt.style.use("science")


def freq(wavelength: float):
    return constants.c / wavelength * 1e9

# test test


def residue(f, y, yerr):
    """Berechnet das gewichtete Residuum eines Datenfits f"""
    return np.sum((f - y)**2 / 1)


def rvalue(f, x, y):
    """Berechnet das Bestimmtheitsmaß der Fit-Funktion f"""
    x = np.array(x)
    y = np.array(y)
    mean = y.mean()
    return np.sum((f(x) - mean)**2) / np.sum((y - mean)**2)


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
    "365": 1000,
    "405": 750,
    "436": 600,
    "546": 200,
    "578": 200,
}

U0_data: dict[list] = {}
h = []


# Geraden-Fit
def linear(b, x):
    m = b[0]
    b = b[1]
    return m * x + b


result_file = open("results", "w")

# Dict mit Latextabellen
tex_tables = {}


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

        # Erstelle eine Latex Tabelle
        table_data = [list(data[2]), (list(data[0]), list(data[1]))]

        # Berechne die Wurzel von I sowie den dazugehörigen Fehler und ziehe den Anodenstrom ab
        data[0] = data[0].apply(lambda x: np.sqrt(x - data[0].min()))
        ind = data[0] > 0
        data = pd.DataFrame([data[0][ind], data[1][ind], data[2][ind]])
        data = data.transpose()
        data[1] = data[1] / data[0]

        # Setze Tabelle fort
        table_data.append((list(data[0]), list(data[1])))
        textable = Textable(caption=f"$({sheet[-1]})\,\SI{{{wavelength}}}{{nm}}$", seperator=",")
        textable.add_header(
            r"$U / \unit{\milli\volt}$",
            r"$I / \unit{\pico\ampere}$",
            r"$\sqrt{I-I_0} / \unit{\pico\ampere\tothe{1/2}}$")
        textable.add_values(*table_data)
        if tex_tables.get(wavelength):
            tex_tables[wavelength].append(textable)
        else:
            tex_tables[wavelength] = [textable]

        # Filtert alle Werte raus, wo die kennlinie einen linearen Anstieg hat
        ind = data[2] < u_max_linear[sheet[:3]]

        # Berechne die Ausgleichsgerade und speicher die grenzspannung in U0 ab
        odr_data = odr.RealData(data[2][ind], data[0][ind], sy=data[1][ind])
        model = odr.Model(linear)
        fit = odr.ODR(odr_data, model, beta0=[1, 1])
        output = fit.run()

        # Ergebnisse
        intercept, intercept_err = output.beta[1], output.sd_beta[1]
        slope, slope_err = output.beta[0], output.sd_beta[0]

        # Güte des Fits
        chi_square = functions.chisquare(
            linear, data[2][ind], data[0][ind], data[1][ind], output.beta)
        rval = rvalue(lambda x: linear([slope, intercept], x),
                      data[2][ind], data[0][ind])
        result_file.write(f"""
-----{sheet}----------
residual variance: {output.res_var}
chi square: {chi_square}
r_value: {rval}\n""")

        # Berechne die Grenzspannung
        u0: float = - intercept / slope
        u0_err: float = np.sqrt((intercept_err/slope) **
                                2 + (intercept*slope_err/slope**2)**2)

        # beachte nur die ersten beiden kennlinien pro wellenlänge
        if sheet[-2:] == "_1" or sheet[-2:] == "_2":
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

"""Berechne zuerst die Mittelwerte der berechneten Grenzspannungen und 
fitte diese Wert in abhängigkeit mit der Frequenz. Bestimme daraus das Wirkungsquantum und 
die Austrittsarbeit der Anode"""
U0 = []
for i in U0_data:
    data = [[u[0] for u in U0_data[i]], [u[1] for u in U0_data[i]]]
    u = np.array(data[0]).mean()
    u_err = np.array(data[1]).mean() + np.array(data[0]).std()
    U0.append([freq(float(i)), u, u_err])

U0: pd.DataFrame = pd.DataFrame(U0)
for i in range(1, 3):
    U0[i] = U0[i].map(lambda x: x / 1000)  # Umrechnung in V

# Berechne Geraden-Fit
odr_data = odr.RealData(U0[0]*1e-14, U0[1], sy=U0[2])
model = odr.Model(linear)
fit = odr.ODR(odr_data, model, beta0=[1, 1])
output = fit.run()

# Speicher fit güte
goodness = f"""
-------Photoeffekt-------
residual variance = {output.res_var}
"""

result_file.write(goodness)

slope = (output.beta[0]*1e-14, output.sd_beta[0]*1e-14)
intercept = (output.beta[1], output.sd_beta[1])

planck_constant = (slope[0] * constants.q, slope[1] * constants.q)
work_function = (-intercept[0], intercept[1])

plt.subplots()
plt.errorbar(U0[0], U0[1], yerr=U0[2], marker="o", ms=5, linestyle="")
plt.plot(U0[0], U0[0]*slope[0] + intercept[0],
         label=f"h = {functions.error_round(planck_constant[0], planck_constant[1], 'scientific')[0]} Js")
plt.legend()
plt.ylabel("$U_0$ / V")
plt.xlabel(r"$\nu$ / Hz")
plt.savefig("../figs/photozelle_wirkungsquantum.png", dpi=200)

res_const = f"""
h = {functions.error_round(planck_constant[0], planck_constant[1], 'scientific')[0]} Js
Referenzwert: h = {constants.h} Js

W = {functions.error_round(work_function[0], work_function[1], 'scientific')[0]} eV
"""
print(res_const)
result_file.write(res_const)

result_file.close()

# Erstelle Latex Tabellen
for wavelength in tex_tables:
    texfile = Texfile(f"{wavelength}", "../latex/tabellen/")
    table0 = tex_tables[wavelength][0]
    table1 = tex_tables[wavelength][1]
    texfile.add(table0.make_figure(table1))
    texfile.make()

# Tabellen für hohe Intensität
texfile = Texfile(f"365_high", "../latex/tabellen/")
table0 = tex_tables["365"][2]
table1 = tex_tables["365"][3]
texfile.add(table0.make_figure(table1))
texfile.make()
