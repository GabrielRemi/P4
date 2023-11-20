import os 
import numpy as np
import matplotlib.pyplot as plt
import file_management as fm
from monke import plots

mainpath = os.path.dirname(__file__)
figpath = f"{mainpath}/../protokoll/figs/"
tabpath = f"{mainpath}/../protokoll/tabellen_figuren/"
os.chdir(mainpath)

t = [120, 10] # Messdauer in sekunden pro Messwinkel
def do_fits() -> dict[str, fm.FitResult]:
    """Macht alle Fits und speichert die Daten in einer Struktur"""
    result = {}
    
    data: list[fm.FileData] = fm.read_file("feinstruktur.txt")
    
    for i, spectrum in enumerate(data):
        sd_y = np.sqrt(spectrum.data[1] / t[i])
        sd_x = [0.01]*len(spectrum.data[0])
        spectrum.add_error(sd_y)
        spectrum.add_error(sd_x)
        spectrum.run_fits()
        result[spectrum.name] = spectrum.fitresult
        
        fig, ax = plt.subplots()
        ax.set_xlim(spectrum.plot_interval)
        ax.errorbar(*spectrum.data, marker="o", ms=3, linestyle="")
        ax.scatter(spectrum.fitresult.x0[0], spectrum.fitresult.height[0], color="red")
        for fit in spectrum.result:
            output = spectrum.result[fit]
            data = output.get_fit_data(output.file_interval.interval, 100)
            ax.plot(*data, label=output.file_interval.name)
            
        plots.legend(ax)
        
        fig.savefig(f"{figpath}{spectrum.name[:-4]}.pdf", dpi=200)
    plt.close()
    return result