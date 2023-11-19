"""Auswertung der Aufgabe 2 (Material Analyse)"""
import os
import numpy as np
from monke import plots
import matplotlib.pyplot as plt
from file_management import read_file
from dataclasses import dataclass
# from scienceplots import scienceplots

mainpath = os.path.dirname(__file__)
figpath = f"{mainpath}/../protokoll/figs/"
os.chdir(mainpath)

@dataclass
class MaAnResult:
    """Speichert alle Ergebnisse eines Spektrums in dieser Klasse"""
    x0: np.ndarray = None
    std: np.ndarray = None
    m: np.ndarray = None
    n: np.ndarray = None
    a: np.ndarray = None
    chi_squared: np.ndarray = None

# Zerstörungsfreie Materialanalyse

def main() -> dict[str, MaAnResult]:
    results: dict[str, MaAnResult] = {}
    """Main Funktion"""
    zma_file = open("material-analyse-ergebnisse.txt", "w", encoding="UTF-8")

    filedata_zm = read_file("material-analyse.txt")
    for file in filedata_zm:
        results[file.name] = MaAnResult()
        x0 = [[], []]
        std = [[], []]
        m = [[] ,[]]
        n = [[], []]
        a = [[], []]
        chi_squared = []
        
        zma_file.write(f"{file.name}\n")
        error = np.sqrt(file.data[1])
        error_x = [1]*len(file.data[0])
        file.add_error(error)
        file.add_error(error_x)
        _, ax = plt.subplots()
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
                dic = out.result[fit_in_out]
                
                if "gauss" in fit_in_out:
                    x0[0].append(dic["x0"][0])
                    x0[1].append(dic["x0"][1])
                    std[0].append(dic["std"][0])
                    std[1].append(dic["std"][1])
                    a[0].append(dic["amplitude"][0])
                    a[1].append(dic["amplitude"][1])
                    m[0].append(0)
                    m[1].append(0)
                    chi_squared.append(0)
                                    
                    text = f"       {fit_in_out}"
                    text += f" x0 = {out.result[fit_in_out]["x0"]}"
                    text += f" std = {out.result[fit_in_out]["std"]}"
                    text += f" a = {out.result[fit_in_out]["amplitude"]}"
                    text += "\n"
                    zma_file.write(text)
                if "linear" in fit_in_out:
                    m[0][-1] = dic["slope"][0]
                    m[1][-1] = dic["slope"][1]
                    
                    text = f"       {fit_in_out}"
                    text += f" n = {out.result[fit_in_out]["intercept"]}"
                    text += f" m = {out.result[fit_in_out]["slope"]}"
                    text += "\n"
                    zma_file.write(text)
                if "chi" in fit_in_out:
                    chi_squared[-1] = dic
                    
                    text = f"   chi_squared = {out.result[fit_in_out]}\n"
                    zma_file.write(text)
            zma_file.write("\n")
            
            results[file.name].x0 = np.array(x0)
            results[file.name].std = np.array(std)
            results[file.name].a = np.array(a)
            results[file.name].m = np.array(m)
            results[file.name].chi_squared = np.array(chi_squared)
            results[file.name].n = np.array(n)
            
        for i, j in plot_data:
            ax.plot(*i, label=j)
        plots.legend(ax)
        plt.savefig(f"{figpath}{file.name[:-4]}.pdf", dpi=200)
    zma_file.close()
    
    return results
