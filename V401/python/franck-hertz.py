import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots.styles
import gauss_fits_franck_hertz as gffh


plt.style.use("science")
plt.rcParams["figure.figsize"] = (7, 5.5)


def do_gauss_fits() -> dict[str, pd.DataFrame]:
    """Gauss Fits mit Diagrammen"""
    data: dict[str, pd.DataFrame] = {}
    data_path_rel: str = "../Data/Franck-Hertz/"

    for file in os.listdir(data_path_rel):
        data[file] = pd.DataFrame(np.loadtxt(f"{data_path_rel}{file}", skiprows=1), columns=["U1", "U2"])


    output = gffh.do_gauss_fits()

    output["x0"].to_csv("Frank-Hertz.csv")
    colors: dict[str, str] = {
        "t_165_u2_2.5.txt": "tab:red",
        "t_165_u2_3.0.txt": "tab:cyan",
        "t_165_u2_3.5.txt": "tab:green",
        "t_165_u2_4.0.txt": "royalblue",
        "u2_2.5_t_165.txt": "tab:red",
        "u2_2.5_t_172.txt": "tab:cyan",
        "u2_2.5_t_179.txt": "tab:green",
        "u2_2.5_t_186.txt": "royalblue"
    }

    for i in data:
        # plt.ylim((-.2, 1))
        plt.xlim((9, 40.65))
        if "t_165_u2" in i:
            plt.figure(1)
            plt.title(r"$T = 165^\circ$")
            plt.errorbar(data[i].U1, data[i].U2, yerr=output[f"{i} error"], ms=3,
                         label=f"{i[-7: -4]} eV $\\chi^2$ = {round(output["chi_squared"][i], 2)}",
                         color=colors[i], marker="o", linestyle="")
            data_key = f"{i} fit_data"
            if data_key in output.keys():
                plt.plot(output[data_key]["x"], output[data_key]["y"],
                         color=colors[i])
            plt.legend(loc="upper left")

        if "u2_2.5_t_" in i:
            plt.figure(2)

            plt.title(r"$U_2 = 2.5\,\mathrm{eV}$")
            plt.errorbar(data[i].U1, data[i].U2, yerr=output[f"{i} error"], ms=3,
                         label=f"${i[-7: -4]}\\,^\\circ$C $\\chi^2$ = {round(output["chi_squared"][i], 2)}",
                         color=colors[i], marker="o", linestyle="")
            data_key = f"{i} fit_data"
            if data_key in output.keys():
                plt.plot(output[data_key]["x"], output[data_key]["y"],
                         color=colors[i])
            plt.legend(loc="upper left")
        plt.xlabel("Beschleunigungsspannung / V")
        plt.ylabel("Anodenstrom / V")
    plt.figure(1)
    plt.savefig("../figs/franck-hertz_gegenspannung.pdf", dpi=200)
    plt.figure(2)
    plt.savefig("../figs/franck-hertz_temperatur.pdf", dpi=200)
    plt.show()

    return output


if __name__ == "__main__":
    data = do_gauss_fits()

    columns = data["x0"].columns.tolist()
    delta_energy = pd.DataFrame(data["x0"].iloc[1:].to_numpy() - data["x0"].iloc[:-1].to_numpy(),
                                columns=columns)
    delta_energy_error = pd.DataFrame(
        np.sqrt(data["x0_err"].iloc[1:].to_numpy()**2 + data["x0_err"].iloc[:-1].to_numpy()**2),
        columns=columns)

    delta_energy_description = delta_energy.describe()
    delta_energy_description.loc["std"] += delta_energy_error.mean()

    delta_energy = pd.concat([delta_energy, delta_energy_description.loc["mean": "std"]])
    delta_energy.to_csv("delta_energy_franck_hertz.csv")
    plt.close()