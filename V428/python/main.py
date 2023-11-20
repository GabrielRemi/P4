import material_analyse
import matplotlib.pyplot as plt
from scienceplots import scienceplots
from typing import Callable, Tuple

plt.style.use("science")
plt.rcParams["figure.figsize"] = [8, 6.5]
plt.rcParams["lines.markersize"] = 3

output = material_analyse.do_fits()
material_analyse.gauss_fit_table(output["FeZn.txt"], "FeZn_tabelle")
callibration: Callable[[float], Tuple[float, float]] = material_analyse.callibrate_energies(output["FeZn.txt"])
print(callibration(100))
