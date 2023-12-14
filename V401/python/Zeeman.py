import numpy as np
from file_management import read_file, FileData
from fit import Fit
import os
import matplotlib.pyplot as plt
from monke import plots, functions
from typing import Tuple
from dataclasses import dataclass

python_path = os.path.dirname(__file__)

etalon_width: float = 4e-3  # in meter


@dataclass
class Zeeman:
    delta_lambda_left: np.ndarray
    delta_lambda_right: np.ndarray
    current: np.ndarray


def do_gauss_fits() -> Zeeman:
    """führt alle Gauss Fits durch und gibt im Array die Winkel der Fit
    Schwerpunkte wieder"""
    os.chdir(python_path)

    delta_lambda_left: list[float] = []
    delta_lambda_right: list[float] = []
    current: list[float] = []

    fits: list[FileData] = read_file("zeeman.txt")
    for fit in fits:
        fit.add_error([1] * len(fit.data[0, :]))
        fit.run_fits()

        fig, ax = plt.subplots()
        ax.set_xlim(fit.plot_interval)
        ax.errorbar(*fit.data, ms=3, linestyle="", marker="o")
        for i in fit.result:
            out: Fit = fit.result[i]
            data = out.get_fit_data(out.file_interval.interval, 200)
            ax.plot(*data, label=out.file_interval.name)

        x0 = fit.fitresult.x0
        delta_lambda_left.append(
            2*etalon_width*(np.cos(x0[0, 1]) - np.cos([x0[0, 0]])))
        delta_lambda_right.append(
            2 * etalon_width * (np.cos(x0[0, 2]) - np.cos([x0[0, 1]])))

        current.append(float(fit.name[2:-4]))

        print(f"{fit.name}: {x0} {fit.fitresult.a}")
        plots.legend(ax)

        ax.plot()
        plt.show()
        print(delta_lambda_left, delta_lambda_right)

    return Zeeman(np.ndarray(delta_lambda_left), np.ndarray(delta_lambda_right), np.ndarray(current))

do_gauss_fits()
