import os 
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import file_management as fm
from monke import plots, functions, latex
import monke.constants as con

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
        ax.set_xlabel(r"Targetwinkel / $^\circ$")
        ax.set_ylabel(r"Mittlere Zählrate / $\mathrm{s}^{-1}$")
        ax.set_xlim(spectrum.plot_interval)
        ax.errorbar(*spectrum.data, marker="o", ms=3, linestyle="", label="Messdaten", color="black")
        ax.scatter(spectrum.fitresult.x0[0], spectrum.fitresult.height[0], color="red")
        for fit in spectrum.result:
            output = spectrum.result[fit]
            data = output.get_fit_data(output.file_interval.interval, 100)
            ax.plot(*data, label=f"{output.file_interval.name} mit $\\chi^2 = {round(output.chi_squared, 2)}$", zorder=100)
            
        plots.legend(ax)
        
        fig.savefig(f"{figpath}{spectrum.name[:-4]}.pdf", dpi=200)
    plt.close()
    return result

def calc_nlambda(deg: float, deg_error: float) -> list[float]:
    """Berechnet Ordnung * Wellenlänge bei gegebenem Messwinkel"""
    
    d = 564 # in pm
    nl = d*np.sin(deg*np.pi/180)
    nl_error = d*np.cos(deg*np.pi/180)*deg_error*np.pi/180
    return [nl, nl_error]

def calc_energies(wavelength: float, wavelength_error: float) -> list[float]:
    """Berechnet aus gegebener Wellenlänge (in pm!) die Energie in keV des Lichts"""
    energy = con.h * con.c / wavelength / con.q * 1e+9
    energy_error = energy * wavelength_error/wavelength
    return [energy, energy_error]

def do_feinstruktur_analysis(output: fm.FitResult) -> None:
    """Berechnet Feinstruktur aus Messung und erstellt Tabellen"""
    os.chdir(mainpath)
    
    degrees, degrees_error = output.x0
    wavelengths = np.array(calc_nlambda(degrees, degrees_error)) / 4 # vierte beugungsordnung
    energies = np.array(calc_energies(*wavelengths))
    wavelength_difference = wavelengths[0][1] - wavelengths[0][0]
    wavelength_difference_error = np.sqrt(wavelengths[1][1]**2 + wavelengths[1][0]**2) 
    wl_reference = np.array([[70.9328, 71.3612], [0.0022, 0.0025]])
    e_reference = np.array([[17.47910, 17.37418], [0.00055, 0.00062]])
    wavelength_difference_reference = wl_reference[0][1] - wl_reference[0][0]
    wavelength_difference_reference_error = np.sqrt(wl_reference[1][1]**2 + wl_reference[1][0]**2) 
    
    ### Speicher Daten in Textdatei
    with open("results", "a", encoding="UTF-8") as file:
        file.write("-------FEINSTRUKTUR----------\n")
        for vals in zip(degrees, degrees_error, *wavelengths, *energies):
            text = f"winkel = {vals[0]}+-{vals[1]}, "
            text += f"wellenlänge = {vals[2]}+-{vals[3]} pm, "
            text += f"E = {vals[4]}+-{vals[5]} keV\n"
            file.write(text)
        text = f"delta l = {wavelength_difference} +- {wavelength_difference_error} pm\n"
        text += f"delta l ref = {wavelength_difference_reference} +- {wavelength_difference_reference_error} pm\n"
        file.write(text)
        
    ### Tabelle 
    with latex.Texfile("feinstruktur_tabelle", tabpath) as file:
        table = latex.Textable(
            "Bestimmung der Wellenlängen und Energien aus der Messung der Feinstrukturaufspaltung mit Referenzwerten aus \\cite{nist_xray_database}",
            "tab:feinstruktur",
            caption_above=True)
        table.add_header(
            r" Winkel $\vartheta/ \unit{\degree}$",
            r"Wellenlänge $\lambda / \unit{\pm}$",
            r"Literaturwert $\lambda / \unit{\pm}$",
            r"Energie $E / \unit{\kilo\electronvolt}$ ",
            r"Literaturwert $E / \unit{\kilo\electronvolt}$ "
        )
        table.add_values(
            (degrees, degrees_error),
            (wavelengths[0], wavelengths[1]),
            (wl_reference[0], wl_reference[1]),
            (energies[0], energies[1]),
            (e_reference[0], e_reference[1])
        )
        file.add(table.make_figure())
    
def do_anode_analysis(output: fm.FitResult) -> None:
    """Führe Auswertung der Molybdän Anode durch"""
    ind = np.argsort(output.x0[0])
    degrees = np.array([output.x0[0][ind], output.x0[1][ind]])
    nl = np.array(calc_nlambda(*degrees))
    ne = np.array(calc_energies(*nl))
    
    ## Berechne Beugungsordnung 
    n = []
    n_err = []
    n1 = nl[0][0]
    n2 = nl[0][1]
    ind1 = [True, False, True, False, True, False, False]
    for i, j in enumerate(ind1):
        if j:
            x = nl[0][i] / n1
            x1 = nl[1][i] / nl[0][i]
            x2 = nl[1][0] / nl[0][0]
            xerr = x*np.sqrt(x1**2 + x2**2)
            n.append(x)
            n_err.append(xerr)
        else:
            x = nl[0][i] / n2
            x1 = nl[1][i] / nl[0][i]
            x2 = nl[1][1] / nl[0][1]
            xerr = x*np.sqrt(x1**2 + x2**2)
            n.append(x)
            n_err.append(xerr)
            
    
    with latex.Texfile("unbekannte_anode", tabpath) as file:
        table = latex.Textable(
            "Bestimmung der Wellenlängen und Energien pro Beugungsordnung aus dem gemessenen Spektrum der unbekannten Anode",
            "tab:anode-unbekannt",
            caption_above=True
        )
        table.add_header(
            r" Winkel $\vartheta/ \unit{\degree}$",
            r"Wellenlänge $n\lambda / \unit{\pm}$",
            r"Energie $En^{-1} / \unit{\kilo\electronvolt}$ "
        )
        table.add_values(
            (degrees[0], degrees[1]),
            (nl[0], nl[1]),
            (ne[0], ne[1])
        )
        file.add(table.make_figure())
    
def main():
    """Auswertung zur Aufgabe 1"""
    output = do_fits()
    do_feinstruktur_analysis(output["feinstruktur.txt"])
    do_anode_analysis(output["anode-unbekannt.txt"])