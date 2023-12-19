import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots.styles
plt.style.use("science")
plt.rcParams["figure.figsize"] = (7, 5.5)

x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
z = [.1, .2, .3, .4]
a = [.5, .6, .7, .8]

b = list(zip([list(a) for a in zip(x, y)], [list(b) for b in zip(z, a)]))

def logp(t: float | np.ndarray):
    return 10.55 - 3333/t - 0.85*np.log(t)

if __name__ == "__main__":
    plt.figure(1)
    x = np.linspace(0, 600, 1000)
    plt.xlabel(r"$T / {}^\circ\mathrm C$")
    plt.ylabel(r"$P$ / Torr")

    plt.plot(x - 273.15, np.exp(logp(x)), linewidth=2)
    plt.savefig("../figs/druck_temp.pdf", dpi=200)
