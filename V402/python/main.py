import pandas as pd
import scipy
import monke
import numpy as np
import matplotlib.pyplot as plt
import venv

excel_file = pd.ExcelFile("../Daten/photozelle_kennlinie.xlsx")
data = pd.read_excel(excel_file, "365_1")


