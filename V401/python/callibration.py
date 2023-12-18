import os
import matplotlib.pyplot as plt
from scipy import odr, optimize
from typing import Callable, Tuple
import numpy as np
import pandas as pd
from monke import functions, plots, constants, latex
import scienceplots.styles

filepath = os.path.dirname(__file__)
os.chdir(filepath)
print(os.getcwd())

plt.style.use("science")
plt.rcParams["figure.figsize"] = [7, 5]
plt.rcParams["figure.dpi"] = 150

func = Callable[[float], float]


def fit_function(x: float, a: float, b: float, c: float, d: float, e: float) -> float:
    """Fit Funktion für das Magnetfeld nach der oben beschriebenen Formel"""
    denom = (c + np.exp(-d * x)) ** e

    return a + b / denom


def do_calibration(data: pd.DataFrame, sy) -> Tuple[list[float], list[float], func, float]:
    """Führe mit scipy.optimize eine Funktionsanpassung an den Datensatz durch"""
    x = data["I / A"]
    y = data["B / mT"]

    popt, pcov = optimize.curve_fit(fit_function, x, y)

    chi_squared = functions.chisquare(lambda b, x: fit_function(x, *b), x, y, sy, popt)

    return popt, pcov, lambda x: fit_function(x, *popt), round(chi_squared, 2)


def main() -> Tuple[func, func]:
    """Main Funktion der Magnetfeldkalibration. gibt die Kalibrationsfunktionen
    vor und nach der Hauptmessung wieder."""
    # Lese die Daten ein
    data_before: pd.DataFrame = pd.DataFrame(np.loadtxt("../Data/Zeeman/kalibrierung_3.txt", skiprows=1),
                                             columns=["I / A", "B / mT"]).query("`I / A` < 9.5")
    data_after: pd.DataFrame = pd.DataFrame(np.loadtxt("../Data/Zeeman/kalibrierung_ende.txt", skiprows=1),
                                            columns=["I / A", "B / mT"]).query("`I / A` < 9.5")
    # data_after.plot(kind="scatter", x="I / A", y="B / mT")

    data_std: float = round(np.std(data_before[341:363]["B / mT"]), 1)

    # Führe Kalibration durch
    beta_before, sd_beta_before, bfield_before, chi_before = do_calibration(data_before, data_std)
    beta_after, sd_beta_after, bfield_after, chi_after = do_calibration(data_after, data_std)
    sd_beta_before: list[float] = np.sqrt(np.diag(sd_beta_before))
    sd_beta_after: list[float] = np.sqrt(np.diag(sd_beta_after))

    # Plotte die Fits
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$I / \mathrm{A}$")
    ax.set_ylabel(r"$B / \mathrm{mT}$")

    ax.errorbar(data_before["I / A"], data_before["B / mT"], yerr=data_std,
                linestyle="", ms=3, marker="o", label="Messung vorher", color="tab:cyan")
    ax.plot(data_before["I / A"], bfield_before(data_before["I / A"]), label="Anpassung", color="tab:cyan")

    ax.errorbar(data_after["I / A"], data_after["B / mT"], yerr=data_std,
                linestyle="", ms=3, marker="o", label="Messung nachher", color="tab:green")
    ax.plot(data_after["I / A"], bfield_after(data_after["I / A"]), label="Anpassung", color="tab:green")

    plots.legend(ax)
    fig.savefig("../figs/BFeld-Kalibrationskurve.pdf", dpi=200)
    # plt.show()
    plt.close()

    # Tabelle
    with latex.Texfile("kallibration_tabelle", "../protokoll/tabellen/") as file:
        table: latex.Textable = latex.Textable(
            "Anpassparameter der Magnetfeldkalibration",
            "fig:kalibration_tabelle", caption_above=True)
        values = list(zip([list(a) for a in zip(beta_before, beta_after)],
                          [list(a) for a in zip(sd_beta_before, sd_beta_after)]))
        table.fig_mode = "h"
        table.add_header(
            " Magnetfeldmessung",
            r"$\alpha / \unit{\milli\tesla}$",
            r"$\beta / \unit{\milli\tesla}$",
            r"$\gamma$",
            r"$\delta$",
            r"$\epsilon$",
            r"$\chi^2$"
        )
        table.add_values(
            ["vorher", "nachher"],
            *values, [chi_before, chi_after])
        file.add(table.make_figure())

    return bfield_before, bfield_after
