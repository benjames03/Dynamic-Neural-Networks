import os
import numpy as np
import matplotlib.pyplot as plt

dirpath = "../results/faults/"
files = [file for file in os.listdir(dirpath) if "summary" not in file and ".txt" in file]
files.sort(key=lambda x: int(x[:-4]))

labels = ["Accuracy", "Prediction margin"]
results = [[], []]
for filepath in files:
    temp = np.loadtxt(dirpath + filepath, delimiter=",").T
    results[0].append(temp[0])
    results[1].append(temp[1])

fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
for i in range(len(results)):
    axs[i].boxplot(results[i][0:-1])
    axs[i].set_xlabel("Number of Faults")
    axs[i].set_ylabel(labels[i])
    axs[i].set_xticklabels(range(0, len(results[i])-1))
    axs[i].axhline(y=results[i][0].mean(), color="blue", linestyle="--", linewidth=1, label="0 fault")

plt.tight_layout()
plt.show()