import os 
import sys
import numpy as np
from scipy import odr
from monke import constants, functions, latex
import pandas as pd
import matplotlib.pyplot as plt
from fit import Fit
from file_management import FileData, read_file


os.chdir(os.path.dirname(__file__))
filedata = read_file("fit_daten.txt")
for file in filedata:
    file.add_error([0.2]*len(file.data[0]))
    file.run_fits()

    data = file.data
    fig, ax = plt.subplots()
    ax.errorbar(data[0], data[1], yerr=data[2])
    for fit in file.result.values():
        interval = (np.min(file.data[0]), np.max(file.data[0]))
        plot_data = fit.get_fit_data(interval, 200)
        plt.plot(*plot_data)
    plt.show()
