import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

COLOURS = ["blue", "red", "green", "purple"]

data = np.load("../results/exp_data.npy", allow_pickle=True)
conf_matrix = np.load("../results/conf_matrix.npy", allow_pickle=True)

fig, (ax1, ax3, ax5) = plt.subplots(1, 3, figsize=(16, 4))
ax2 = ax1.twinx()
ax4 = ax3.twinx()

fig_manager = plt.get_current_fig_manager()
fig_manager.set_window_title("Big-Little Analysis")

# Accuracy and loss graph
ax1.set_title("Accuracy and Loss over Threshold")
ax1.plot(data[0]["thresholds"], data[0]["accuracies"], label="ResNet9 Accuracy", color=COLOURS[0], marker="o", ms=3)
ax2.plot(data[0]["thresholds"], data[0]["losses"], label="ResNet9 Loss", color=COLOURS[1], marker="o", ms=3)
ax1.plot(data[1]["thresholds"], data[1]["accuracies"], label="ResNet20 Accuracy", color=COLOURS[2], marker="o", ms=3)
ax2.plot(data[1]["thresholds"], data[1]["losses"], label="ResNet20 Loss", color=COLOURS[3], marker="o", ms=3)
ax1.legend(loc="upper center")#, bbox_to_anchor=(1, 0.65))
ax2.legend(loc="lower center")#, bbox_to_anchor=(1, 0.45))

ax1.set_xlabel("Threshold")
ax1.set_ylabel("Accuracy")#, color=COLOURS[0])
ax2.set_ylabel("Loss")#, color=COLOURS[1])

# ax2.spines["left"].set_color(COLOURS[0])
# ax1.tick_params(axis="y", colors=COLOURS[0])
# ax2.spines["right"].set_color(COLOURS[1])
# ax2.tick_params(axis="y", colors=COLOURS[1])

# time and operations graph
ax3.set_title("Time and Operations over Threshold")
ax3.plot(data[0]["thresholds"], data[0]["cpu_time"], label="ResNet9 Time", color=COLOURS[0], marker="o", ms=3)
ax4.plot(data[0]["thresholds"], data[0]["operations"], label="ResNet9 Operations", color=COLOURS[1], marker="o", ms=3)
ax3.plot(data[1]["thresholds"], data[1]["cpu_time"], label="ResNet20 Time", color=COLOURS[2], marker="o", ms=3)
ax4.plot(data[1]["thresholds"], data[1]["operations"], label="ResNet20 Operations", color=COLOURS[3], marker="o", ms=3)
ax3.legend(loc="upper left", bbox_to_anchor=(0, 0.8))
ax4.legend(loc="upper left", bbox_to_anchor=(0, 1))

ax3.set_xlabel("Threshold")
ax3.set_ylabel("Execution Time (s)")#, color=COLOURS[2])
ax4.set_ylabel("No. Operations (TFLOPS)")#, color=COLOURS[3])

# ax4.spines["left"].set_color(COLOURS[2])
# ax3.tick_params(axis="y", colors=COLOURS[2])
# ax4.spines["right"].set_color(COLOURS[3])
# ax4.tick_params(axis="y", colors=COLOURS[3])

# confusion matrix heatmap
ax5.set_title("Confusion Matrix Heatmap")
ax6 = sns.heatmap(conf_matrix, annot=False, ax=ax5, cmap="Oranges", cbar=True)
ax5.set_xlabel("Predicted Label")
ax5.set_ylabel("True Label")
cbar = ax6.collections[0].colorbar
cbar.set_label("No. Correct")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.4)
plt.show()