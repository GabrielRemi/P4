"""Dieses Script führt die Gauss Fits aus und gibt die Winkel der Peaks wieder"""

import numpy as np
import pandas as pd
import os
from file_management import read_file, FileData, FitResult
from fit import Fit
import matplotlib.pyplot as plt
from monke import plots, functions
from typing import Tuple
from dataclasses import dataclass
import scienceplots.styles

python_path = os.path.dirname(__file__)
plt.style.use("science")
plt.rcParams["figure.figsize"] = [7, 5.5]


def do_gauss_fits() -> pd.DataFrame:
    """führt alle Gauss Fits durch und gibt im Array die Winkel der Fit
    Schwerpunkte wieder. Alle Winkel werden in Radianten wiedergegeben"""
    os.chdir(python_path)

    result: pd.DataFrame = pd.DataFrame([])
    current: list[float] = []
    theta_left: list[float] = []
    sd_theta_left: list[float] = []
    theta_middle: list[float] = []
    sd_theta_middle: list[float] = []
    theta_right: list[float] = []
    sd_theta_right: list[float] = []

    fits: list[FileData] = read_file("zeeman.txt")
    for fit in fits:
        current.append(float(fit.name[1:4]))

        error = np.array([np.sqrt(i) if np.sqrt(i) > 1 else 1 for i in fit.data[1]])  # Fehler unskaliert
        fit.add_error(error / 3)
        fit.run_fits()
        fitres: FitResult = fit.fitresult
        if len(fitres.x0[0]) != 3:
            print(fitres.x0)
            raise Exception("Fehler: im Gauss Fit wurden keine 3 Gausskurven gefittet")

        fig, ax = plt.subplots()
        ax.set_xlabel(r"$\theta\,/\,^\circ$")
        ax.set_ylabel(r"Intensität $I\,/\,\%$")

        ax.set_xlim(fit.plot_interval)
        ax.errorbar(*fit.data[:2], yerr=fit.data[2], ms=3, linestyle="", marker="o", label="Messwerte")
        for i in fit.result:
            out: Fit = fit.result[i]
            data = out.get_fit_data(out.file_interval.interval, 200)
            ax.plot(*data, label="Gauss-Anpassung")

        plots.legend(ax)

        # ax.plot()
        os.chdir(python_path)
        fig.savefig(f"../figs/gauss_{fit.name[:4]}.pdf", dpi=200)
        # plt.show()

        theta_left.append(fitres.x0[0, 0] * np.pi / 180)
        sd_theta_left.append(fitres.x0[1, 0] * np.pi / 180)
        theta_middle.append(fitres.x0[0, 1] * np.pi / 180)
        sd_theta_middle.append(fitres.x0[1, 1] * np.pi / 180)
        theta_right.append(fitres.x0[0, 2] * np.pi / 180)
        sd_theta_right.append(fitres.x0[1, 2] * np.pi / 180)


    result["I"] = current
    result["deg_l"] = theta_left
    result["deg_l_err"] = sd_theta_left
    result["deg_m"] = theta_middle
    result["deg_m_err"] = sd_theta_middle
    result["deg_r"] = theta_right
    result["deg_r_err"] = sd_theta_right
    return result


def main():
    data = do_gauss_fits()

    data.to_csv("gauss_fits_zeeman.csv")


if __name__ == "__main__":
    main()
