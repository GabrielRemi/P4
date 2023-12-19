import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
z = [.1, .2, .3, .4]
a = [.5, .6, .7, .8]

b = list(zip([list(a) for a in zip(x, y)], [list(b) for b in zip(z, a)]))

if __name__ == "__main__":
    x = ["2", "1", "3"]
    x.sort()
    print(x)
