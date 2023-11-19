import os
import shlex
import numpy as np
from fit import Fit, DataInterval

class FileData:
    """Eine Klasse mit einem Datensatz sowie allen Intervallen,
    in denen fits durchgeführt werden soll"""

    def __init__(self, name):
        self.file_intervals: list[DataInterval] = []
        self.data: np.ndarray = np.array([])
        self.__read_data(name)
        self.result: dict[str, Fit] = None
        self.name = name

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


def read_file(name: str) -> list[FileData]:
    """Liest die Datei, in der zu jedem Datensatz die Anleitung für die Gauß fits steht"""
    file_data: list[FileData] = []

    with open(name, encoding="UTF-8") as file:
        for line in file.readlines():
            args = shlex.split(line)
            if len(args) == 0:
                continue

            if args[0] == "dir":
                os.chdir(args[1])
            elif args[0] == "begin":
                file_data.append(FileData(args[1]))
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
