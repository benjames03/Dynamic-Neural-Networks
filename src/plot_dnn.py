import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLOURS = ["blue", "red", "green", "purple"]

data = np.load("../results/exp_data.npy", allow_pickle=True)
conf_matrix = np.load("../results/conf_matrix.npy", allow_pickle=True)

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(14, 4))

fig_manager = plt.get_current_fig_manager()
fig_manager.set_window_title("Big-Little Analysis")

# ax1.set_title("Accuracy over Threshold")
# ax1.set_xlabel("Threshold")
# ax1.set_ylabel("Accuracy")
# ax1.plot(data[1]["thresholds"], data[1]["accuracies"], label="ResNet20", color=COLOURS[0], marker="o", ms=3)
# # ax1.legend()
# ax1.grid(True)

# ax3.set_title("Loss over Threshold")
# ax3.set_xlabel("Threshold")
# ax3.set_ylabel("Loss")
# ax3.plot(data[1]["thresholds"], data[1]["losses"], label="ResNet20", color=COLOURS[1], marker="o", ms=3)
# # ax3.legend()
# ax3.grid(True)

ax1.set_title("Execution Time over Threshold")
ax1.set_xlabel("Threshold")
ax1.set_ylabel("Execution Time (s)")
ax1.plot(data[1]["thresholds"], data[1]["cpu_time"], label="ResNet20", color=COLOURS[2], marker="o", ms=3)
# ax1.legend()
ax1.grid(True)

ax3.set_title("Number of Operations over Threshold")
ax3.set_xlabel("Threshold")
ax3.set_ylabel("Number of Operations (TFLOPS)")
ax3.plot(data[1]["thresholds"], data[1]["operations"], label="ResNet20", color=COLOURS[3], marker="o", ms=3)
# ax3.legend()
ax3.grid(True)

plt.tight_layout()
plt.savefig("../results/graphs/dynn_comp_time.pdf", format="pdf", bbox_inches="tight")
plt.show()
# cpu_time operations accuracies losses