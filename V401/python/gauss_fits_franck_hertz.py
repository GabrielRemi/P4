import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots.styles
from file_management import read_file, FileData
from fit import Fit
from typing import Tuple, Callable

python_path = os.path.dirname(__file__)
plt.style.use("science")
plt.rcParams["figure.figsize"] = [7, 5.5]


def do_gauss_fits() -> dict[str, pd.DataFrame | np.ndarray]:
    os.chdir(python_path)

    x0: list[list[float]] = []
    x0_err: list[list[float]] = []
    sigma: list[list[float]] = []
    sigma_err: list[list[float]] = []
    chi_squared: list[list[float]] = []

    x0_df: pd.DataFrame = pd.DataFrame()
    sigma_df: pd.DataFrame = pd.DataFrame()
    chi_squared_df: pd.DataFrame = pd.DataFrame()
    x0_err_df: pd.DataFrame = pd.DataFrame()
    sigma_err_df: pd.DataFrame = pd.DataFrame()

    result: dict[str, pd.DataFrame | np.ndarray] = {}
    #fit_data: pd.DataFrame = pd.DataFrame()

    fits: list[FileData] = read_file("franck-hertz.txt")
    for fit in fits:
        # Y Fehler
        for dat in reversed(fit.data):
            error = []
            for elem in dat:
                e = 0.01*elem
                if elem < 3:
                    e += 0.015
                elif elem < 10:
                    e += .05
                elif elem < 30:
                    e += .15
                else:
                    e += .5
                error.append(e)
            error = np.array(error)
            fit.add_error(error)

        fit.run_fits()

        out = fit.fitresult

        x0_df[fit.name] = out.x0[0]
        x0_err_df[fit.name] = out.x0[1]
        sigma_df[fit.name] = out.std[0]
        sigma_err_df[fit.name] = out.std[1]
        chi_squared_df[fit.name] = out.chi_squared

        # print(fit.result["Name"].file_interval.interval)
        # fit_data["x"], fit_data[fit.name] = (
        #     fit.result["Name"].get_fit_data(fit.result["Name"].file_interval.interval, 400))

        result[f"{fit.name} fit_data"] = pd.DataFrame(
            fit.result["Name"].get_fit_data(fit.result["Name"].file_interval.interval, 400).transpose(),
            columns=["x", "y"])
        result[f"{fit.name} error"] = fit.data[2]
        result[f"{fit.name} error x"] = fit.data[3]

    result["x0"] = x0_df
    result["x0_err"] = x0_err_df
    result["sigma"] = sigma_df
    result["sigma_err"] = sigma_err_df
    result["chi_squared"] = chi_squared_df.iloc[-1]

    os.chdir(python_path)
    return result

if __name__ == "__main__":
    out = do_gauss_fits()
    for i in out[0]:
        print(i)
        print(out[0][i])

    print(out[1])