import os 
import sys
import numpy as np
from scipy import odr
from monke import constants, functions, latex
import pandas as pd
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(__file__))

data = np.loadtxt("../data/V428/Charakteristisches Spektrum/feinstruktur.txt", skiprows=1)
data = np.loadtxt("../data/V428/Zerstörungsfreie Materialanalyse/Ag1.txt", skiprows=1)

fig, ax = plt.subplots()
ax.plot(data[:,0], data[:,1])
plt.show()

print(type(lambda x: x))