import os
import numpy as np
from monke import functions as fct
from monke.latex import Texfile, Textable

os.chdir(os.path.dirname(__file__))

data4nm = np.loadtxt("../data/HOPG10597.csv", skiprows=1)
data2nm = np.loadtxt("../data/HOPG10628.csv", skiprows=1, delimiter=",")

r4nm = data4nm[:, 3]  # in pm
phi4nm = data4nm[:, 2]  # in deg
r2nm = data2nm[:, 3]
phi2nm = data2nm[:, 2]

n = list(range(1, len(r4nm) + 1))

table = Textable(
    r"relative Winkel zur Horizontalen und eingezeichnete Abstände aus \cref{fig:hopg_rtm_4nm_1_cur,fig:hopg_rtm_4nm_2_cur}",
    caption_above=True,
    label="tab:atomabstand")
table.add_header(
    f" Linie",
    r"$\varphi_1 / {}^\circ\pm \SI{10}{\degree}$",
    r"$r_1/\unit{\pm} \pm \SI{90}{\pm}$",
    r"$\varphi_2 / {}^\circ\pm \SI{10}{\degree}$",
    r"$r_2/\unit{\pm} \pm \SI{100}{\pm}$")
table.add_values(
    n,
    [abs(int(round(i, -1))) for i in phi4nm],
    [int(round(i, -1)) for i in r4nm],
    [abs(int(round(i, -1))) for i in phi2nm],
    [int(round(i, -2)) for i in r2nm]
)


print(f"r_mean 4nm = {np.mean(r4nm)} pm +- {np.std(r4nm) + 90} pm")
print(f"r_mean 2nm = {np.mean(r2nm)} pm +- {np.std(r2nm) + 100} pm")

with Texfile("HOPG10597", "../protokoll/tabellen/") as file:
    file.add(table.make_figure())
