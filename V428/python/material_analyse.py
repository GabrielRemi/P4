"""Auswertung der Aufgabe 2 (Material Analyse)"""
import os
import sys
import numpy as np
from scipy import odr
from monke import constants, functions, latex, plots
import pandas as pd
import matplotlib.pyplot as plt
from fit import Fit
from file_management import FileData, read_file
# from scienceplots import scienceplots

mainpath = os.path.dirname(__file__)
figpath = f"{mainpath}/../protokoll/figs/"
os.chdir(mainpath)

# plt.style.use("science")


# Zerstörungsfreie Materialanalyse


def main():
    """Main Funktion"""
    zma_file = open("material-analyse-ergebnisse.txt", "w")

    filedata_zm = read_file("material-analyse.txt")
    for file in filedata_zm:
        zma_file.write(f"{file.name}\n")
        error = np.sqrt(file.data[1])
        error_x = [1]*len(file.data[0])
        file.add_error(error)
        file.add_error(error_x)
        fig, ax = plt.subplots()
        ax.set_xlim(file.plot_interval)
        ax.errorbar(*file.data, marker="x", linestyle="")

        # Intervalle
        file.run_fits()
        plot_data = []
        for i in file.result:
            out = file.result[i]
            plot_data.append((out.get_fit_data(out.file_interval.interval, 200),
                              out.file_interval.name))
            
            ## in txt datei speichern
            zma_file.write(f"   {out.file_interval.name}\n")
            for fit_in_out in out.result:
                if "gauss" in fit_in_out:
                    text = f"       {fit_in_out}"
                    text += f" x0 = {out.result[fit_in_out]["x0"]}"
                    text += f" std = {out.result[fit_in_out]["std"]}"
                    text += f" a = {out.result[fit_in_out]["amplitude"]}"
                    text += "\n"
                    zma_file.write(text)
                if "linear" in fit_in_out:
                    text = f"       {fit_in_out}"
                    text += f" n = {out.result[fit_in_out]["intercept"]}"
                    text += f" m = {out.result[fit_in_out]["slope"]}"
                    text += "\n"
                    zma_file.write(text)
                if "chi" in fit_in_out:
                    text = f"   chi_squared = {out.result[fit_in_out]}\n"
                    zma_file.write(text)
            zma_file.write("\n")
            
        for i, j in plot_data:
            ax.plot(*i, label=j)
        plots.legend(ax)
        plt.savefig(f"{figpath}{file.name[:-4]}.pdf", dpi=200)
    zma_file.close()
