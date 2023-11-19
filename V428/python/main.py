import material_analyse
import matplotlib.pyplot as plt
from scienceplots import scienceplots

plt.style.use("science")
plt.rcParams["figure.figsize"] = [8, 6.5]
plt.rcParams["lines.markersize"] = 3

output = material_analyse.main()
print(output["FeZn.txt"].x0[0])
