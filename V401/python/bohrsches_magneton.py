"""Mithilfe der Gauss-Fits und der Kalibrationskurven wird hier das bohrsche Magneton ermittelt.
Hierzu wird zunächst aus den Winkelmessungen die Energieverschiebung bestimmt
und aus einem Geradenfit bei Auftragung gegen das B-Feld die Steigung, welche dem Magneton entspricht."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import odr
from typing import Callable

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
    m = 2*d*n/wavelength * np.cos(data["deg_m"])
    m, m_err = round(m.mean(), 0), m.std()

    data["bfield_before"] = bfield_before(data["I"])
    data["bfield_after"] = bfield_after(data["I"])

    data["delta_l_left"] = 2*n*d/m * (np.cos(data.deg_m) - np.cos(data.deg_l))
    data["delta_l_left_err"] = 2*n*d/m * np.sqrt(
        (np.sin(data.deg_m)*data.deg_m_err)**2
        + (np.sin(data.deg_l)*data.deg_l_err)**2)
    data["delta_l_right"] = 2*n*d/m * (np.cos(data.deg_r) - np.cos(data.deg_m))
    data["delta_l_right_err"] = 2*n*d/m * np.sqrt(
        (np.sin(data.deg_m)*data.deg_m_err)**2
        + (np.sin(data.deg_r)*data.deg_r_err)**2)

    data["energy_left"] = energy_0 * data.delta_l_left / (wavelength - data.delta_l_left)
    data["energy_left_err"] = data["energy_left"] * data.delta_l_left_err / data.delta_l_left
    data["energy_right"] = energy_0 * data.delta_l_right / (wavelength + data.delta_l_right)
    data["energy_right_err"] = data["energy_right"] * data.delta_l_right_err / data.delta_l_right
    data["bfield"] = (data.bfield_before + data.bfield_after)/2
    data["bfield_err"] = abs(data.bfield_after - data.bfield)

    return data


def linear(b, x):
    return b[1] * x + b[0]


def do_fit(data: pd.DataFrame) -> pd.DataFrame:
    """erstelle einen Fit, um das Magneton zu bestimmen"""
    model: odr.Model = odr.Model(linear)
    fitdata: odr.RealData = odr.RealData(x=data.bfield, y=data.energy_left*1e21, sx=data.bfield_err,
                                         sy=data.energy_left_err*1e21)
    fit = odr.ODR(model=model, data=fitdata, beta0=[0.0, 9.27e-6])
    out: odr.Output = fit.run()
    left_vals = [out.beta[0]* 1e-21, out.sd_beta[0]* 1e-21,
                 out.beta[1] * 1e-18, out.sd_beta[1] * 1e-18]  # mal 1000 damit magneton in J / T

    fitdata: odr.RealData = odr.RealData(x=data.bfield, y=data.energy_right, sx=data.bfield_err,
                                         sy=data.energy_right_err)
    fit = odr.ODR(model=model, data=fitdata, beta0=[0, 9.2e-6])
    out: odr.Output = fit.run()
    right_vals = [out.beta[0], out.sd_beta[0],
                 out.beta[1] * 1e-18, out.sd_beta[1] * 1e-18]  # mal 1000 damit magneton in J / T

    return pd.DataFrame([left_vals, right_vals],
                        columns=["Achsenabschnitt n", "n Fehler", "Magneton", "Magneton Fehler"],
                        index=["Linker Peak", "Rechter Peak"])


def plot(data: pd.DataFrame, fitresult: pd.DataFrame):
    """Plotte die Fits"""
    fig, ax = plt.subplots()
    ax.errorbar(data.bfield, data.energy_left, xerr=data.bfield_err, yerr=data.energy_left_err,
                ms=3, marker="o", linestyle="", label="Messwerte")

    fit: Callable[[float], float] = lambda x: linear([fitresult["Achsenabschnitt n"]["Linker Peak"],
                                                      fitresult["Magneton"]["Linker Peak"]], x)
    ax.plot(data.bfield, fit(data.bfield), label="Fit")

    plt.show()

if __name__ == "__main__":
    data: pd.DataFrame = calc_data_for_plotting()
    fitresult = do_fit(data)
    print(fitresult)
    plot(data, fitresult)