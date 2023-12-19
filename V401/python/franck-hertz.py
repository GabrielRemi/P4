import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots.styles
import gauss_fits_franck_hertz as gffh
from monke import latex


plt.style.use("science")
plt.rcParams["figure.figsize"] = (7, 5.5)


def do_gauss_fits() -> dict[str, pd.DataFrame]:
    """Gauss Fits mit Diagrammen"""
    data: dict[str, pd.DataFrame] = {}
    data_path_rel: str = "../Data/Franck-Hertz/"

    for file in os.listdir(data_path_rel):
        data[file] = pd.DataFrame(np.loadtxt(f"{data_path_rel}{file}", skiprows=1), columns=["U1", "U2"])


    output: dict[str, pd.DataFrame] = gffh.do_gauss_fits()

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
        #plt.ylim((-.2, 1))
        plt.xlim((9, 42))
        if "t_165_u2" in i:
            plt.figure(1)
            plt.title(r"$T = 165^\circ$")
            plt.errorbar(data[i].U1, data[i].U2, yerr=output[f"{i} error"], ms=3,
                         xerr=output[f"{i} error x"],
                         label=f"$U_\\mathrm{{G}} =$ {i[-7: -4]} V, $\\chi^2$ = {round(output["chi_squared"][i], 2)}",
                         color=colors[i], marker="o", linestyle="")
            data_key = f"{i} fit_data"
            if data_key in output.keys():
                plt.plot(output[data_key]["x"], output[data_key]["y"],
                         color=colors[i])
            plt.legend(loc="upper left")

        if "u2_2.5_t_" in i:
            plt.figure(2)

            plt.title(r"$U_\mathrm G = 2.5\,\mathrm{eV}$")
            plt.errorbar(data[i].U1, data[i].U2, yerr=output[f"{i} error"], ms=2,
                         xerr=output[f"{i} error x"],
                         label=f"$T = {i[-7: -4]}\\,^\\circ$C $\\chi^2$ = {round(output["chi_squared"][i], 2)}",
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
    #plt.show()

    ## Erstelle Tabellen
    keys = ["x0", "x0_err", "sigma", "sigma_err"]
    columns = output["x0"].columns.tolist()
    table_data: dict[str, pd.DataFrame] = {}
    for curve in columns:
        data: list[list[float]] = []
        for key in keys:
            data.append(output[key][curve].tolist())
        #print(data)
        table_data[curve] = pd.DataFrame(np.array(data).transpose(), columns=keys)

    # for i in table_data.keys():
    #     print(i, table_data[i], sep="\n", end="\n\n")

    with latex.Texfile("franck_hertz_tabellen", "../protokoll/tabellen/") as file:
        ## Erste Tabelle für Gegenspannung
        keys = list(filter(lambda x: "t_165_u2" in x, table_data.keys()))
        keys.sort()
        preheader: list[str] = list(map(lambda x: f"\\multicolumn{{2}}{{|c}}{{\\SI{{{x[-7:-4]}}}{{\\volt}}}}", keys))
        assert(len(keys) == 4)

        table: latex.Textable = latex.Textable(
            r"Anpassparameter der Spannungskurve für verschiedene Gegenspannungen",
            label="tab:gegenspannung", caption_above=True)
        table.fig_mode = "htb"
        table.alignment = "c|cc|cc|cc|cc"
        table.add_line_before_header("", *preheader)
        table.add_hline()
        table.add_header("Maximum", *([r"$x_0$ / \unit{\volt}", r"$\sigma$ / \unit{\volt}"]*4))
        values = []
        for key in keys:
            values.extend([(table_data[key].x0, table_data[key].x0_err),
                          (table_data[key].sigma, table_data[key].sigma_err)])
        table.add_values(list(range(1,7)), *values)
        file.add(table.make_figure())

        ## Zweite Tabelle für Temperatur
        keys = list(filter(lambda x: "u2_2.5_t" in x, table_data.keys()))
        keys.sort()
        preheader: list[str] = list(map(lambda x: f"\\multicolumn{{2}}{{|c}}{{\\SI{{{x[-7:-4]}}}{{\\celsius}}}}", keys))
        assert (len(keys) == 4)

        table: latex.Textable = latex.Textable(
            r"Anpassparameter der Spannungskurve für verschiedene Temperaturen",
            label="tab:temperatur", caption_above=True)
        table.fig_mode = "htb"
        table.alignment = "c|cc|cc|cc|cc"
        table.add_line_before_header("", *preheader)
        table.add_hline()
        table.add_header("Maximum", *([r"$x_0$ / \unit{\volt}", r"$\sigma$ / \unit{\volt}"] * 4))
        values = []
        for key in keys:
            values.extend([(table_data[key].x0, table_data[key].x0_err),
                           (table_data[key].sigma, table_data[key].sigma_err)])
        table.add_values(list(range(1, 7)), *values)
        file.add(table.make_figure())

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

    # Tabellen:
    with latex.Texfile("delta_energy_franck_hertz_tabellen", "../protokoll/tabellen/") as file:
        n = [*list(range(1,6)), r"$\langle E\rangle$"]
        table_gegen = latex.Textable("Übergangsenergien bei verschiedenen Gegenspannungen in eV",
                                     "fig:energy_gegen", caption_above=True)
        table_gegen.fig_mode = "htb"
        keys = list(filter(lambda key: "t_165_u2" in key, delta_energy.columns))
        keys.sort()
        header = list(map(lambda key: f"\\SI{{{key[-7:-4]}}}{{\\volt}}", keys))

        values = [n]
        table_gegen.add_header(" Maximum", *header)
        for key in keys:
            values.append(([*delta_energy[:5][key], delta_energy.loc["mean"][key]],
                           [*delta_energy_error[key], delta_energy.loc["std"][key]]))
        table_gegen.add_values(*values)


        table_temp = latex.Textable("Übergangsenergien bei verschiedenen Temperaturen in eV",
                                    "fig:energy_temp", caption_above=True)
        keys = list(filter(lambda key: "u2_2.5_t" in key, delta_energy.columns))
        keys.sort()

        values = [n]
        header = list(map(lambda key: f"\\SI{{{key[-7:-4]}}}{{\\celsius}}", keys))
        table_temp.fig_mode = "htb"
        table_temp.add_header(" Maximum", *header)
        for key in keys:
            values.append(([*delta_energy[:5][key], delta_energy.loc["mean"][key]],
                           [*delta_energy_error[key], delta_energy.loc["std"][key]]))
        table_temp.add_values(*values)

        file.add(table_gegen.make_figure())
        file.add(table_temp.make_figure())



    #print(delta_energy[:5], delta_energy_error, sep="\n")