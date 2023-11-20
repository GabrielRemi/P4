import numpy as np
import pandas as pd
from monke import latex, functions
import scipy.odr as odr
from dataclasses import dataclass

@dataclass
class DataInterval:
    """Ansammlung von Daten, die für das Fitten im Interval nötig sind"""
    interval: tuple
    n_fits: int
    linear: bool
    name: str
    start_parameters: list[float]

def _fit_function(b: list[float], x: float, n: int = 1, linear=True):
    """the Funktion, an die gefittet werden soll"""
    result = 0
    if linear:
        result += b[3 + 3*(n-1)] + b[4 + 3*(n-1)]*x

    for i in range(n):
        pre = b[0 + 3*i]/(np.sqrt(2*np.pi) * b[2+3*i])
        e = np.exp(-(x - b[1+3*i])**2 / (2*b[2+3*i]**2))
        result += pre*e

    return result


class Fit:
    """speichert Daten eines fits wie name und anzahl der zu benutzenden Gauss-Kurven"""
    def __init__(self, file_interval: DataInterval):
        self.file_interval = file_interval
        self.chi_squared: float = None
        self.parameters: list[float] = None
        self.parameters_std: list[float] = None
        self.result: dict[str, dict[str, float]] = {}

    def do_fit(self, data: np.ndarray):
        """Erstellt mit scipy.odr einen Datenfit im Intervall <interval>.
        <interval> ist ein 2-Element tuple, Falls ein linearer Offset besteht, so muss
        lin=True gesetzt werden."""
        def fit_function(b: list[float], x: float) -> float:
            return _fit_function(b, x, self.file_interval.n_fits, self.file_interval.linear)
        #print(f"types: {type(self.file_interval.interval[0])}, {type(data[0][0])}")
        ind = [bool(self.file_interval.interval[0] <= item <=
                    self.file_interval.interval[1]) and data[1][i] != 0 for i, item in enumerate(data[0])]

        model: odr.Model = odr.Model(fit_function)
        realdata: odr.RealData = odr.RealData(
            x=data[0][ind], y=data[1][ind], sy=data[2][ind])
        if len(data) == 4:
            realdata: odr.RealData = odr.RealData(
                x=data[0][ind], y=data[1][ind], sy=data[2][ind], sx=data[3][ind])
        myodr = odr.ODR(data=realdata, model=model,
                        beta0=self.file_interval.start_parameters)
        output = myodr.run()
        for n in range(self.file_interval.n_fits):
            self.result[f"gauss {n}"] = {"amplitude": (output.beta[3*n], output.sd_beta[3*n]),
                                         "x0": (output.beta[1 + 3*n], output.sd_beta[1 + 3*n]),
                                         "std": (abs(output.beta[2 + 3*n]), output.sd_beta[2 + 3*n])}
        if self.file_interval.linear:
            self.result["linear"] = {"intercept": (output.beta[-2], output.sd_beta[-2]),
                                     "slope": (output.beta[-1], output.sd_beta[-2])}
        self.parameters = output.beta
        self.parameters_std = output.sd_beta

        self.chi_squared = functions.chisquare(
            fit_function, data[0][ind], data[1][ind], data[2][ind], self.parameters)
        self.result["chi_squared"] = self.chi_squared

    def get_fit_data(self, interval: tuple, n: int) -> np.ndarray:
        """erstelle ein numpy array mit n Elementen in einem intervall mit dem fit als Funktion"""
        fit_x = np.linspace(interval[0], interval[1], n)
        fit_y = _fit_function(self.parameters, fit_x, self.file_interval.n_fits, self.file_interval.linear)
        return np.array([fit_x, fit_y])


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    def gauss(x, x0, s):
        return 1/(np.sqrt(2*np.pi)*s) * np.exp(-(x - x0)**2 / (2*s**2))

    x = np.linspace(0, 20, 90)
    y = 3*gauss(x, 6.5, 0.5) + 1*gauss(x, 5, 0.5) + 0.5 * \
        gauss(x, 15, 0.5) + (np.random.rand(len(x)) - 0.5)*0.3 + 0.07*x
    err = [0.1]*len(x)
    data = np.array([x, y, err, err])
    data[3] = data[3] * 3

    for i in data.transpose():
        print(*i)
    fileint = DataInterval((0, 20), 3, True, "name", [
              2, 7, 0.5, 1, 5, 0.5, 0.5, 15, 0.5, 0, 0.1])
    fit = Fit(fileint)
    fit.do_fit(data)
    print("beta: ", fit.parameters)
    print(fit.result)
    fit_data = fit.get_fit_data((0, 20), 200)

    fig, ax = plt.subplots()
    ax.errorbar(data[0], data[1], marker="x",
                linestyle="", yerr=data[2], xerr=data[3])
    ax.plot(fit_data[0], fit_data[1])
    plt.show()
