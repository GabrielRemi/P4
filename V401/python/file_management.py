import os
from dataclasses import dataclass
import shlex
import numpy as np
from fit import Fit, DataInterval


@dataclass
class FitResult:
    """Speichert alle Ergebnisse eines Spektrums in dieser Datenklasse"""
    x0: np.ndarray = None
    std: np.ndarray = None
    m: np.ndarray = None
    n: np.ndarray = None
    a: np.ndarray = None
    height: np.ndarray = None
    chi_squared: np.ndarray = None


class FileData:
    """Eine Klasse mit einem Datensatz sowie allen Intervallen,
    in denen fits durchgeführt werden soll"""

    def __init__(self, name):
        self.file_intervals: list[DataInterval] = []
        self.data: np.ndarray = np.array([])
        self.__read_data(name)
        self.result: dict[str, Fit] | None = None
        self.name = name
        self.plot_interval = None
        self.fitresult: FitResult = FitResult()

    def __read_data(self, file):
        """Liest Daten aus einer Datei ein"""
        self.data = np.loadtxt(file, skiprows=1).transpose()

    def add_error(self, array: np.ndarray | list[float]):
        """fügt zu den Daten eine weitere zeile hinzu die den fehler darstellen soll"""
        self.data = np.array([*self.data, array])

    def _add_file_intervals(self, args: list[str]):
        """Fügt ein Interval zum Datensatz hinzu"""
        linear = True if args[3] in ["True", "true"] else False
        interval = DataInterval(
            (float(args[0]), float(args[1])), int(args[2]), linear, args[4], [float(i) for i in args[5:]])
        self.file_intervals.append(interval)

    def run_fits(self):
        """führt für alle Intervalle die fits durch"""
        self.result = dict()
        for interval in self.file_intervals:
            if len(self.data) < 3:
                raise Exception("Cannot do fit because no errors were given")
            fit = Fit(interval)
            fit.do_fit(self.data)
            self.result[interval.name] = fit
            
        self.__init_fitresult()

    def __init_fitresult(self):
        """initiiert self.fitresult nachdem alles fits gemacht wurden. fitresult
        ist eine alternative Speichermethode der fitdaten im vergleich 
        zu self.result"""
        x0 = [[], []]
        std = [[], []]
        m = [[], []]
        n = [[], []]
        a = [[], []]
        chi_squared = []
        
        for i in self.result:
            out = self.result[i]
            
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
                    n[0].append(0)
                    n[1].append(0)
                    chi_squared.append(0)
                if "linear" in fit_in_out:
                    m[0][-1] = dic["slope"][0]
                    m[1][-1] = dic["slope"][1]
                    n[0][-1] = dic["intercept"][0]
                    n[1][-1] = dic["intercept"][1]
                if "chi" in fit_in_out:
                    chi_squared[-1] = dic

        self.fitresult.x0 = np.array(x0)
        self.fitresult.std = np.array(std)
        self.fitresult.a = abs(np.array(a))
        self.fitresult.m = np.array(m)
        self.fitresult.chi_squared = np.array(chi_squared)
        self.fitresult.n = np.array(n)
        self.fitresult.height = np.array(np.array([
            self.fitresult.a[0] / (np.sqrt(2*np.pi) * self.fitresult.std[0]),
            self.fitresult.a[0] / (np.sqrt(2*np.pi) * self.fitresult.std[0]) *
            np.sqrt((self.fitresult.a[1]/self.fitresult.a[0])**2 +
                    (self.fitresult.std[1]/self.fitresult.std[0])**2)
        ]))


def read_file(name: str) -> list[FileData]:
    """Liest die Datei, in der zu jedem Datensatz die Anleitung für die Gauß fits steht.
    Das Einlesen einer Datei sieht wie folgt aus:\n
    begin Datei.txt interval_min interval_max
        i_min i_max n_fits linear? name A x0 std A x0 std ... n m
        ...
    end"""
    file_data: list[FileData] = []

    with open(name, encoding="UTF-8") as file:
        for line in file.readlines():
            args = shlex.split(line)
            if len(args) == 0:
                continue
            if args[0][0] == "#":
                continue

            if args[0] == "dir":
                os.chdir(args[1])
            elif args[0] == "begin":
                file_data.append(FileData(args[1]))
                if len(args) >= 4:
                    file_data[-1].plot_interval = (float(args[2]),
                                                   float(args[3]))
                else:
                    dmin = file_data[-1].data[0][0]
                    dmax = file_data[-1].data[0][-1]
                    file_data[-1].plot_interval = (dmin, dmax)
            elif args[0] == "end":
                continue
            else:
                file_data[-1]._add_file_intervals(args)

    return file_data


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    filedata = read_file("fit_daten.txt")
    for file in filedata:
        file.add_error([np.sqrt(len(file.data[0]))]*file.data[0])
        file.run_fits()
        print(file.result)
