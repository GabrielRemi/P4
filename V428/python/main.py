from typing import Callable, Tuple
import matplotlib.pyplot as plt
from scienceplots import scienceplots
import material_analyse
from material_analyse import Callibration
import bragg_reflexion

plt.style.use("science")
plt.rcParams["figure.figsize"] = [6, 4.5]
plt.rcParams["lines.markersize"] = 3

with open("results", "w", encoding="UTF-8") as file:
    pass

bragg_reflexion.main()

def aufgabe2():
    """Auswertung zur Aufgabe 2"""
    output = material_analyse.do_fits()
    material_analyse.gauss_fit_table(output["FeZn.txt"], "FeZn_tabelle")

    callibration: Callibration = material_analyse.callibrate_energies(output["FeZn.txt"])
    material_analyse.make_callibration_table(output, callibration)

    for n in range(1, 5):
        material_analyse.make_callibration_table_for_one(output[f"Unbekannt{n}.txt"], callibration, f"Unbekannt{n}")
    
    material_analyse.mass_fractions(output)

aufgabe2()
