"""Mithilfe der Gauss-Fits und der Kalibrationskurven wird hier das bohrsche Magneton ermittelt.
Hierzu wird zunächst aus den Winkelmessungen die Energieverschiebung bestimmt
und aus einem Geradenfit bei Auftragung gegen das B-Feld die Steigung, welche dem Magneton entspricht.
Linker Peak->Rechtszirkular, rechter Peak->Linkszirkular"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr
from typing import Callable, Tuple

from monke import functions, plots
from monke import constants as cnst

import callibration
import gauss_fits_zeeman as gfz

n = 1.457
d = 4e-3  # Etalon Breite
wavelength = 643.8e-9  # Wellenlänge Laser
energy_0 = cnst.h*cnst.c / wavelength  # in J


def calc_data_for_plotting() -> pd.DataFrame:
    """Berechnet alle nötigen Daten, um anschließend einen Datenfit erstellen zu können.
    unter bfield und bfield_err ist das Magnetfeld gespeichert und unter energy_left/right
    respektive die Energieverschiebungen für linken und rechten peak"""
    bfield_before, bfield_after = callibration.main()
    data: pd.DataFrame = pd.read_csv("gauss_fits_zeeman.csv")

    ## Bestimme aus Laser Wellenlänge die Beugungsordnung
    # m = 2*d*n/wavelength * np.cos(data["deg_m"])
    m = 2*d/wavelength * np.sqrt(n**2 - np.sin(data.deg_m)**2)
    m, m_err = round(m.mean(), 0), m.std()
    print(f"Beugungsordnung: {m}")

    def calc_wavelength(x: float) -> float:
        """Berechnet die Wellenlänge bei Beugungsordnung m und Winkel x"""
        return 2*d/m * np.sqrt(n**2 - np.sin(x)**2)

    def calc_wavelength_err(x: float, xerr: float) -> float:
        """Berechnet den zur Wellenlänge dazugehörigen Fehler"""
        prefac = 2 * d / m
        ins = np.sqrt(n ** 2 - np.sin(x))
        return prefac * x*xerr/ins

    def calc_delta_l_err(x: float, xerr: float, y: float, yerr: float) -> float:
        """Berechnet den Fehler der Wellenlängendifferenz"""
        return np.sqrt(calc_wavelength_err(x, xerr)**2 + calc_wavelength_err(y, yerr)**2)

    data["bfield_before"] = bfield_before(data["I"])
    data["bfield_after"] = bfield_after(data["I"])

    data["delta_l_left"] = calc_wavelength(data.deg_l) - calc_wavelength(data.deg_m)
    data["delta_l_left_err"] = calc_delta_l_err(data.deg_l, data.deg_l_err, data.deg_m, data.deg_m_err)
    data["delta_l_right"] = calc_wavelength(data.deg_r) - calc_wavelength(data.deg_m)
    data["delta_l_right_err"] = calc_delta_l_err(data.deg_r, data.deg_r_err, data.deg_m, data.deg_m_err)

    data["energy_left"] = - energy_0 * data.delta_l_left / (wavelength + data.delta_l_left)
    data["energy_left_err"] = abs(data["energy_left"] * data.delta_l_left_err / data.delta_l_left)
    data["energy_right"] = - energy_0 * data.delta_l_right / (wavelength + data.delta_l_right)
    data["energy_right_err"] = abs(data["energy_right"] * data.delta_l_right_err / data.delta_l_right)
    data["bfield"] = (data.bfield_before + data.bfield_after)/2
    data["bfield_err"] = abs(data.bfield_after - data.bfield)

    return data


def linear(b, x):
    return b[1] * x + b[0]


def do_fit(data: pd.DataFrame) -> pd.DataFrame:
    """erstelle einen Fit, um das Magneton zu bestimmen"""
    new_data: pd.DataFrame = data.loc[:, ["bfield", "bfield_err", "energy_left", "energy_left_err",
                                          "energy_right", "energy_right_err"]]
    new_data.iloc[:, 0:2] /= 1000  # BFeld in T
    new_data.iloc[:, 2:] *= 1e24  # Damit Werte in Größenordnung 1 sind

    def do_one_fit(x, y, sx, sy) -> odr.Output:
        model: odr.Model = odr.Model(linear)
        data: odr.RealData = odr.RealData(x, y, sx, sy)
        fit = odr.ODR(model=model, data=data, beta0=[0.0, 9.27])
        return fit.run()

    left_out: odr.Output = do_one_fit(new_data.bfield, new_data.energy_left,
                                      new_data.bfield_err, new_data.energy_left_err)
    right_out: odr.Output = do_one_fit(new_data.bfield, new_data.energy_right,
                                       new_data.bfield_err, new_data.energy_right_err)

    result = pd.DataFrame([[*left_out.beta, *left_out.sd_beta], [*right_out.beta, *right_out.sd_beta]],
                        columns=["Achsenabschnitt", "Magneton", "Achsenabschnitt Fehler", "Magneton Fehler"],
                        index=["linker Peak", "rechter Peak"])
    result.loc[:, ["Magneton", "Magneton Fehler"]] *= 1e-24
    result.loc[:, ["Achsenabschnitt", "Achsenabschnitt Fehler"]] *= 1e-24
    return result

def plot(data: pd.DataFrame, fitresult: pd.DataFrame):
    """Plotte die Fits"""
    plt.figure(1, figsize=(10, 5))
    plt.subplot(121)
    plt.xlabel(r"$B\,/\,\mathrm{mT}$")

    #fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.ylabel(r"$\Delta E\,/\mathrm{J}$")

    plt.errorbar(data.bfield, data.energy_left, xerr=data.bfield_err, yerr=data.energy_left_err,
                ms=3, marker="o", linestyle="", label=r"Messwerte $\sigma^+$")

    # Im plot sind mT statt T angegeben, deshalb m / 1000
    params_left = [fitresult["Achsenabschnitt"]["linker Peak"], fitresult["Magneton"]["linker Peak"]/1000]
    fit_left: Callable[[float], float] = lambda x: linear(params_left, x)

    chi_left = (functions.chisquare(linear, data.bfield, data.energy_left, data.energy_left_err, params_left)
                .__round__(2))

    plt.plot(data.bfield, fit_left(data.bfield), label=f"Anpassung, $\\chi^2 = {chi_left}$")

    params_right = [fitresult["Achsenabschnitt"]["rechter Peak"], fitresult["Magneton"]["rechter Peak"] / 1000]
    fit_right: Callable[[float], float] = lambda x: linear(params_right, x)
    chi_right = (functions.chisquare(linear, data.bfield, data.energy_right, data.energy_right_err, params_right)
                .__round__(2))
    plt.legend()

    plt.subplot(122)
    plt.xlabel(r"$B\,/\,\mathrm{mT}$")
    plt.errorbar(data.bfield, data.energy_right, xerr=data.bfield_err, yerr=data.energy_right_err,
                 ms=3, marker="o", linestyle="", label=r"Messwerte $\sigma^-$")
    plt.plot(data.bfield, fit_right(data.bfield), label=f"Anpassung, $\\chi^2 = {chi_right}$")

    plt.legend()
    plt.savefig("../figs/magneton.pdf", dpi=200)
    plt.show()


    plt.close()

if __name__ == "__main__":
    data: pd.DataFrame = calc_data_for_plotting()
    fitresult = do_fit(data)
    fitresult.loc[:, ["Magneton", "Magneton Fehler"]].abs().to_csv("magneton.csv")
    plot(data, fitresult)