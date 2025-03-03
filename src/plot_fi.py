import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_data(dirpath):
    files = os.listdir(dirpath)
    files.sort(key=lambda x: int(x[:-4]))

    labels = ["Accuracy", "Prediction margin"]
    data = [[], []]
    for filepath in files:
        temp = np.loadtxt(dirpath + filepath, delimiter=",").T
        data[0].append(temp[0])
        data[1].append(temp[1])

    return labels, data

def box_plot(dirpath):
    labels, results = get_data(dirpath)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        data = results[i]
        axs[i].boxplot(data)
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(data)))
        axs[i].set_xticklabels(range(len(data)))
        axs[i].axhline(y=data[0].mean(), color="blue", linestyle="--", linewidth=1, label="0 fault")

    plt.tight_layout()
    plt.show()

def violin_plot(dirpath):
    labels, results = get_data(dirpath)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        data = results[i]
        sns.violinplot(ax=axs[i], data=data, inner="box")
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(data)))
        axs[i].set_xticklabels(range(len(data)))
        axs[i].axhline(y=data[0].mean(), color="blue", linestyle="--", linewidth=1, label="0 fault")

    plt.tight_layout()
    plt.show()

def strip_plot(dirpath):
    labels, results = get_data(dirpath)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        data = results[i]
        sns.stripplot(ax=axs[i], data=data, size=3.5)
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(data)))
        axs[i].set_xticklabels(range(len(data)))
        axs[i].axhline(y=data[0].mean(), color="blue", linestyle="--", linewidth=1, label="0 fault")

    plt.tight_layout()
    plt.show()

def bit_pos_err(dirpath):
    df = pd.read_csv(dirpath, names=["index", "same", "sim", "error"])
    df = df.drop(columns=["same"])
    df["sim"] = df["sim"].astype(float)
    df["error"] = df["error"].str.replace("max err - ", "").astype(float)
    print(df)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df["index"], df["sim"], color="blue", label="Similarity")
    ax1.set_xlabel("Bit position")
    ax1.set_ylabel("Similarity", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")
    ax2 = ax1.twinx()
    ax2.plot(df["index"], df["error"], color="red", label="Max Error")
    ax2.set_xlabel("Bit position")
    ax2.set_ylabel("Max Error", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title("Error over bit positions")
    plt.legend()
    plt.show()

dirpath = "../results/faults_ubiq/"
box_plot(dirpath)

# bit_pos_err("../results/bit_pos_test.txt")