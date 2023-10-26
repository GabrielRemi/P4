import pandas as pd
from scipy import odr
from monke import functions, constants, plots
from monke.latex import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import scienceplots

os.chdir(os.path.dirname(__file__))
plt_params = {
    "font.size": 8.5,
    "lines.markersize": 3,
    "pgf.texsystem": "pdflatex",
    "text.usetex": True,
    "figure.figsize": [7, 5.5]
   
}
mpl.rcParams.update(plt_params)  ## Latex Preambel funktioniert nicht? was ein Schmutz
plt.style.use("science")


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
    "405": 700,
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

# Dict mit Latextabellenwerten
tex_tables = {}

# Dict mit Werten für Latextabelle von geraden-fit-auswertung
textable1 = {
    "365": [],
    "405": [],
    "436": [],
    "546": [],
    "578": [],
}


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
        table_data = [[int(num) for num in data[2]],
                      (list(data[0]), list(data[1]))]

        # Berechne die Wurzel von I sowie den dazugehörigen Fehler und ziehe den Anodenstrom ab
        data[0] = data[0].apply(lambda x: np.sqrt(x - data[0].min()))
        ind = data[0] > 0
        data = pd.DataFrame([data[0][ind], data[1][ind], data[2][ind]])
        data = data.transpose()
        data[1] = data[1] / data[0]

        # Setze Tabelle fort
        # table_data.append((list(data[0]), list(data[1])))
        textable = Textable(caption=f"$({sheet[-1]})\,\SI{{{wavelength}}}{{nm}}$",
                            caption_above=True)
        textable.add_header(
            r"$U / \unit{\milli\volt}$",
            r"$I / \unit{\pico\ampere}$",
            r"$\sqrt{I-I_0} / \unit{\pico\ampere\tothe{1/2}}$")
        textable.add_values(*table_data)
        if tex_tables.get(wavelength):
            tex_tables[wavelength].append(table_data)
        else:
            tex_tables[wavelength] = [table_data]

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

        # Berechne die Grenzspannung
        u0: float = - intercept / slope
        u0_err: float = np.sqrt((intercept_err/slope) **
                                2 + (intercept*slope_err/slope**2)**2)

        result_file.write(f"""
-----{sheet}----------
m = {error_round(slope, slope_err,"parenthesis")}
b = {error_round(intercept, intercept_err, "parenthesis")}
U0 = {error_round(u0, u0_err, "parenthesis")}
residual variance: {output.res_var}
chi square: {chi_square}\n""")

        # beachte nur die ersten beiden kennlinien pro wellenlänge
        texdata = [chi_square, (u0, u0_err)]
        if sheet[-2:] == "_1" or sheet[-2:] == "_2":
            textable1[wavelength].append(texdata)
            if U0_data.get(wavelength):
                U0_data[wavelength].append((u0, u0_err))
            else:
                U0_data[wavelength] = [(u0, u0_err)]

        # Erstelle kennlinien plots
        fig, ax = plt.subplots()
        ax.errorbar(data[2], data[0], yerr=data[1],
                    marker="o", linestyle="", label="Messung")
        xlim = ax.get_xlim()

        x = np.linspace(xlim[0], xlim[1], 4)

        ax.plot(x, intercept + slope * x,
                label=f"$U_0$ = {functions.error_round(u0, u0_err, 'parenthesis')}\,mV")

        ax.set_xlim(xlim)
        ax.set_ylim((-0.5, ax.get_ylim()[1]))
        ax.set_xlabel(r"$U$ / mV")
        ax.set_ylabel(r"$\sqrt{I-I_0}$ / $\text{pA}^{1/2}$")
        plots.legend(ax, size=7)
        plt.savefig(f"../figs/photozelle_kennline_{sheet}.png", dpi=200)

        # Latextabellen

"""Berechne zuerst die Mittelwerte der berechneten Grenzspannungen und 
fitte diese Wert in abhängigkeit mit der Frequenz. Bestimme daraus das Wirkungsquantum und 
die Austrittsarbeit der Anode"""
U0 = []
for i in U0_data:
    data = [[u[0] for u in U0_data[i]], [u[1] for u in U0_data[i]]]
    u = np.array(data[0]).mean()
    u_err = np.array(data[1]).mean() + np.array(data[0]).std()
    U0.append([freq(float(i)), u, u_err])

for (i, elem) in enumerate(textable1):
    textable1[elem].append((U0[i][1], U0[i][2]))

U0: pd.DataFrame = pd.DataFrame(U0)
for i in range(1, 3):
    U0[i] = U0[i].map(lambda x: x / 1000)  # Umrechnung in V

# Berechne Geraden-Fit
odr_data = odr.RealData(U0[0]*1e-14, U0[1], sy=U0[2])
model = odr.Model(linear)
fit = odr.ODR(odr_data, model, beta0=[1, 1])
output = fit.run()


slope = (output.beta[0]*1e-14, output.sd_beta[0]*1e-14)
intercept = (output.beta[1], output.sd_beta[1])

# Speicher fit güte
goodness = f"""
-------Photoeffekt-------
m = {error_round(*slope, "scientific")[0]}
b = {error_round(*intercept, "parenthesis")}
residual variance = {output.res_var}
chi_square = {functions.chisquare(linear, U0[0], U0[1], U0[2], [slope[0], intercept[0]])}
"""

result_file.write(goodness)

planck_constant = (slope[0] * constants.q, slope[1] * constants.q)
work_function = (-intercept[0], intercept[1])

plot_label = f"Anpassungsgerade"
fig, ax = plt.subplots()
ax.errorbar(U0[0], U0[1], yerr=U0[2], marker="o", linestyle="", label="Messung")
ax.plot(U0[0], U0[0]*slope[0] + intercept[0],
         label=plot_label)
ax.set_ylabel("$U_0$ / V")
ax.set_xlabel(r"$\nu$ / Hz")
plots.legend(ax, size=7)
plt.savefig("../figs/photozelle_wirkungsquantum.png", dpi=250)

res_const = f"""
h = {functions.error_round(planck_constant[0], planck_constant[1], 'scientific')[0]} Js
Referenzwert: h = {constants.h} Js

W = {functions.error_round(work_function[0], work_function[1], 'scientific')[0]} eV
"""
print(res_const)
result_file.write(res_const)

result_file.close()
temp_table = None


def kennlinie_table(texdata, name: str, caption: str, temp_table=None, no_output=False) -> Textable | None:
    texfile = Texfile(f"tabelle_{name}", "../latex/tabellen/")
    textable = Textable(caption=caption,
                        caption_above=True,
                        label=caption.lower().replace(" ", "_").replace(
                            r"{", "").replace(r"}", "")
                        .replace("\\si", ""))
    textable.alignment = "cc||cc"
    textable.add_line_before_header(
        "\multicolumn{2}{c||}{erste Messung}",
        "\multicolumn{2}{c}{zweite Messung}")
    textable.add_hline()
    textable.add_header(
        r"$U / \unit{\milli\volt}$",
        r"$I / \unit{\pico\ampere}$",
        r"$U / \unit{\milli\volt}$",
        r"$I / \unit{\pico\ampere}$")
    textable.add_values(*texdata)

    if no_output == True:
        return textable

    if temp_table is None:
        texfile.add(textable.make_figure())
    else:
        texfile.add(textable.make_figure(temp_table))
    texfile.make()


temp_table = None

# Erstelle Latex Tabellen
for wavelength in tex_tables:
    texdata = tex_tables[wavelength][0]
    for l in tex_tables[wavelength][1]:
        texdata.append(l)

    caption = f"Kennlinie \SI{{{wavelength}}}{{nm}}"

    if wavelength == "365":
        kennlinie_table(
            texdata, name=f"{wavelength}", caption=caption)

        texdata = tex_tables[wavelength][2]
        for l in tex_tables[wavelength][3]:
            texdata.append(l)
        kennlinie_table(
            texdata,  name=f"{wavelength}_high", caption=f"{caption} hohe Intensität")
        continue

    if temp_table is None:
        temp_table = kennlinie_table(
            texdata, name=f"{wavelength}", caption=caption, no_output=True)
    else:
        kennlinie_table(
            texdata, name=f"{wavelength}", caption=caption, temp_table=temp_table)
        temp_table = None

# Tabelle für Ausgleichsgeraden
# Erstelle zuerst eine liste mit allen Werten
texdata = [[], [], [], ([], []), ([], []), ([], [])]
for wavelength in textable1:
    data = textable1[wavelength]
    texdata[0].append(int(wavelength))
    texdata[1].append("{:.2f}".format(data[0][0]).replace(".", ","))
    texdata[2].append("{:.2f}".format(data[1][0]).replace(".", ","))
    texdata[3][0].append(data[0][1][0])
    texdata[3][1].append(data[0][1][1])
    texdata[4][0].append(data[1][1][0])
    texdata[4][1].append(data[1][1][1])
    texdata[5][0].append(data[2][0])
    texdata[5][1].append(data[2][1])

file = Texfile("tabelle_grenzspannungen", "../latex/tabellen/")
table = Textable("Bestimmung der Grenzspannungen", "fig:messwerte_grenzspannungen",
                 caption_above=True)
table.add_header(
    r"$\lambda / \unit{\nano\meter}$",
    r"$\chi_1^2/\mathrm{dof}$",
    r"$\chi_2^2/\mathrm{dof}$",
    r"$U_{1,0}/\unit{\milli\volt}$",
    r"$U_{2,0}/\unit{\milli\volt}$",
    r"$U_0/\unit{\milli\volt}$"
)
table.add_values(*texdata)
file.add(table.make_figure())
file.make()

