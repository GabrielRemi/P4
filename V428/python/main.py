import os 
import sys
import numpy as np
from scipy import odr
from monke import constants, functions, latex, plots
import pandas as pd
import matplotlib.pyplot as plt
from fit import Fit
from file_management import FileData, read_file
#from scienceplots import scienceplots

mainpath = os.path.dirname(__file__)
figpath = f"{mainpath}/../protokoll/figs/"
os.chdir(mainpath)

#plt.style.use("science")
plt.rcParams["figure.figsize"] = [8, 6.5]
plt.rcParams["lines.markersize"] = 3

### Zerstörungsfreie Materialanalyse

zma_file = open("material-analyse-ergebnisse.txt", "w")

filedata_zm = read_file("material-analyse.txt")
for file in filedata_zm:
    zma_file.write(f"{file.name}\n")
    file.add_error([1]*len(file.data[0]))
    fig, ax = plt.subplots()
    ax.set_xlim(file.plot_interval)
    ax.errorbar(*file.data, marker="x", linestyle="")
    
    # Intervalle
    file.run_fits()
    plot_data = []
    for i in file.result:
        out = file.result[i]
        zma_file.write(f"   {out.file_interval.name} {out.result}\n")
        plot_data.append((out.get_fit_data(out.file_interval.interval, 200), 
                          out.file_interval.name))

    for i, j in plot_data:
        ax.plot(*i, label=j)
    plots.legend(ax)
    plt.savefig(f"{figpath}{file.name[:-4]}.pdf", dpi=200)
zma_file.close()