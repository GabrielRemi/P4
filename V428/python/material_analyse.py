"""Auswertung der Aufgabe 2 (Material Analyse)"""
import os
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
from monke import plots, latex, functions
import matplotlib.pyplot as plt
from file_management import read_file
import scipy.odr as odr
# from scienceplots import scienceplots

mainpath = os.path.dirname(__file__)
figpath = f"{mainpath}/../protokoll/figs/"
tabpath = f"{mainpath}/../protokoll/tabellen_figuren/"
os.chdir(mainpath)

Callibration = Callable[[float], Tuple[float, float]]


@dataclass
class MaAnResult:
    """Speichert alle Ergebnisse eines Spektrums in dieser Klasse"""
    x0: np.ndarray = None
    std: np.ndarray = None
    m: np.ndarray = None
    n: np.ndarray = None
    a: np.ndarray = None
    height: np.ndarray = None
    chi_squared: np.ndarray = None

# Zerstörungsfreie Materialanalyse


def do_fits() -> dict[str, MaAnResult]:
    """Main Funktion. Liest die Fit Anweisung aus der Datei material-analyse.txt, speichert die 
    Werte in der Datei material-analyse-ergebnisse.txt und gibt in einem dict() alle Ergebnisse wieder"""
    results: dict[str, MaAnResult] = {}
    zma_file = open("material-analyse-ergebnisse.txt", "w", encoding="UTF-8")

    filedata_zm = read_file("material-analyse.txt")
    for file in filedata_zm:
        results[file.name] = MaAnResult()
        x0 = [[], []]
        std = [[], []]
        m = [[], []]
        n = [[], []]
        a = [[], []]
        chi_squared = []

        zma_file.write(f"{file.name}\n")
        error = np.sqrt(file.data[1])
        error_x = [1]*len(file.data[0])
        file.add_error(error)
        file.add_error(error_x)
        _, ax = plt.subplots()
        ax.set_xlim(file.plot_interval)
        ax.errorbar(*file.data, marker="x", linestyle="")

        # Intervalle
        file.run_fits()
        plot_data = []
        for i in file.result:
            out = file.result[i]
            plot_data.append((out.get_fit_data(out.file_interval.interval, 200),
                              out.file_interval.name))

            # in txt datei speichern
            zma_file.write(f"   {out.file_interval.name}\n")
            for fit_in_out in out.result:
                dic = out.result[fit_in_out]

                if "gauss" in fit_in_out:
                    x0[0].append(dic["x0"][0])
                    x0[1].append(dic["x0"][1])
                    std[0].append(dic["std"][0])
                    std[1].append(dic["std"][1])
                    a[0].append(dic["amplitude"][0])
                    a[1].append(dic["amplitude"][1])
                    m[0].append(0)
                    m[1].append(0)
                    n[0].append(0)
                    n[1].append(0)
                    chi_squared.append(0)

                    text = f"       {fit_in_out}"
                    text += f" x0 = {out.result[fit_in_out]["x0"]}"
                    text += f" std = {out.result[fit_in_out]["std"]}"
                    text += f" a = {out.result[fit_in_out]["amplitude"]}"
                    text += "\n"
                    zma_file.write(text)
                if "linear" in fit_in_out:
                    m[0][-1] = dic["slope"][0]
                    m[1][-1] = dic["slope"][1]
                    n[0][-1] = dic["intercept"][0]
                    n[1][-1] = dic["intercept"][1]

                    text = f"       {fit_in_out}"
                    text += f" n = {out.result[fit_in_out]["intercept"]}"
                    text += f" m = {out.result[fit_in_out]["slope"]}"
                    text += "\n"
                    zma_file.write(text)
                if "chi" in fit_in_out:
                    chi_squared[-1] = dic

                    text = f"   chi_squared = {out.result[fit_in_out]}\n"
                    zma_file.write(text)
            zma_file.write("\n")

            results[file.name].x0 = np.array(x0)
            results[file.name].std = np.array(std)
            results[file.name].a = abs(np.array(a))
            results[file.name].m = np.array(m)
            results[file.name].chi_squared = np.array(chi_squared)
            results[file.name].n = np.array(n)
            results[file.name].height = np.array(np.array([
                results[file.name].a[0] /
                (np.sqrt(2*np.pi) * results[file.name].std[0]),
                results[file.name].a[0] / (np.sqrt(2*np.pi) * results[file.name].std[0]) * np.sqrt(
                    (results[file.name].a[1]/results[file.name].a[0])**2 + (results[file.name].std[1]/results[file.name].std[0])**2)
            ]))

        for i, j in plot_data:
            ax.plot(*i, label=j)
        plots.legend(ax)
        plt.savefig(f"{figpath}{file.name[:-4]}.pdf", dpi=200)
    zma_file.close()

    return results


def gauss_fit_table(values: MaAnResult, name: str) -> None:
    """Erstellt für das Protokoll eine Tabelle der Gauss Fits eines Spektrums"""
    os.chdir(mainpath)
    with latex.Texfile(name, tabpath) as file:
        table: latex.Textable = latex.Textable("Gauss-Anpassungen an das FeZn-Spektrum.", "tab:fezn-gauss-fits",
                                               caption_above=True)
        table.add_header(
            r" Digitaler Kanal $K$",
             r"Standardabweichung $\sigma$",
             r"Höhe der Gauß-Kurve $H$",
             r"$\chi^2$"
        )
        table.add_values(
            (values.x0[0], values.x0[1]),
            (values.std[0], values.std[1]),
            (values.height[0], values.height[1]),
            [round(i, 2) if i != 0 else "-" for i in values.chi_squared]
        )
        
        file.add(table.make_figure())

def callibrate_energies(fezn: MaAnResult) -> Callable[[float], Tuple[float, float]]:
    """Berechnet aus FeZn Spektrum die Kallibrationskurve und gibt die Kallibrationskurve wieder"""
    energies = np.array([6403.84, 7057.98, 8638.86, 9572.0])

    # Tabelle mit Kanälen und dazugehörigen Energien
    with latex.Texfile("kallibration_tabelle", tabpath) as file:
        table = latex.Textable("Kanäle mit dazugehörigen Energien zur Kallibration der Kanäle",
                               "tab:callibration", caption_above=True)
        table.add_header(
            r" Digitaler Kanal $K$",
            r"Energie $E / \unit{\electronvolt}$"
        )
        table.add_values(
            (fezn.x0[0], fezn.x0[1]),
            ["{:.2f}".format(i) for i in energies]
        )
        
        file.add(table.make_figure())

    def linear_model(b, x):
        return b[0] + x*b[1]
    model = odr.Model(linear_model)
    data = odr.RealData(fezn.x0[0], energies, sx=fezn.x0[1])
    fit = odr.ODR(data, model, beta0=[0, 1])
    output = fit.run()

    intercept = (output.beta[0], output.sd_beta[0])
    slope = (output.beta[1], output.sd_beta[1])

    def callibration_curve(x):
        value = linear_model([intercept[0], slope[0]], x)
        error = np.sqrt((intercept[1])**2 + (slope[1]*x)**2)
        return (value, error)

    fig, ax = plt.subplots()

    ax.errorbar(fezn.x0[0], energies, xerr=fezn.x0[1],
                linestyle="", marker="x", markersize=6)
    ax.plot(fezn.x0[0], callibration_curve(fezn.x0[0])[0])
    fig.savefig(f"{figpath}kallibrationskurve.pdf", dpi=200)

    return callibration_curve

def make_callibration_table(metals: dict[str, MaAnResult], callibration: Callibration) -> None:
    """Erstelle eine Tabelle, Wo für jedes gemessene Element die Energie peaks
    mit ihrer Höhe eingetragen sind"""
    text: str = r"""
\begin{table}[H]
    \centering
    \caption{gemessene Energie und Höhe der charakteristischen Linien verschieder Metalle}
    \label{tab:tab:energien-charakeristische-linien}
    \begin{tabular}{c|c|c}
        Metall & Energie $E/\unit{\kilo\electronvolt}$ & Höhe in Detektionen \\\hline"""
    
    for metal in metals:
        if "Unbekannt" in metal:
            continue
        vals: MaAnResult = metals[metal]
        energies: np.ndarray = callibration(vals.x0[0])
        #print(functions.error_round(energies[0], energies[1]))
        detections: np.ndarray = vals.height
        n: int = len(energies[0])
        text += f"\\multirow{{{n}}}{{*}}{{{metal.replace(".txt", "")}}}"
        
        for index in range(n):
            energy: Tuple[str] = functions.error_round(energies[0][index]/1000, energies[1][index]/1000)
            detection: Tuple[str] = functions.error_round(detections[0][index], detections[1][index])
            text += f" & \\num{{{energy[0]}\\pm {energy[1]}}} & \\num{{{detection[0]}\\pm {detection[1]}}} \\\\"
        text+= "\\hline\n"

    text += r"""
    \end{tabular}
\end{table}"""

    with open(f"{tabpath}charakteristische_linien.tex", "w", encoding="UTF-8") as file:
        file.write(text)
        
def make_callibration_table_for_one(metal: MaAnResult, callibration: Callibration, name: str):
    """Erstelle eine Tabelle Wo für ein gemessenes Spektrum die Energie peaks
    mit ihrer Höhe eingetragen sind"""
    text: str = f"""
\\begin{{table}}[H]
    \\centering
    \\caption{{Energien der charakteristischen Linien von {name.replace(".txt", "")}}}
    \\label{{tab:label}}
    \\begin{{tabular}}{{c|c}}
       Energie $E/\\unit{{\\kilo\\electronvolt}}$ & Höhe in Detektionen \\\\\n\\hline\n"""
    
    energies: np.ndarray = callibration(metal.x0[0])
    detections: np.ndarray = metal.height
    n: int = len(energies[0])
        
    for index in range(n):
        energy: Tuple[str] = functions.error_round(energies[0][index]/1000, energies[1][index]/1000)
        detection: Tuple[str] = functions.error_round(detections[0][index], detections[1][index])
        text += f"\\num{{{energy[0]}\\pm {energy[1]}}} & \\num{{{detection[0]}\\pm {detection[1]}}} \\\\ \n"

    text += r"""
    \end{tabular}
\end{table}"""

    with open(f"{tabpath}{name}.tex", "w", encoding="UTF-8") as file:
        file.write(text)