import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def get_data(dirpath):
    files = sorted(os.listdir(dirpath), key=lambda x: int(x[:-4]))

    labels = ["Accuracy", "Prediction margin"]
    data = [[], []]
    for filepath in files:
        temp = np.loadtxt(dirpath + filepath, delimiter=",", usecols=(0, 1)).T
        data[0].append(temp[0])
        data[1].append(temp[1])

    return labels, data

def basic_plot(dirpath):
    labels, results = get_data(dirpath)
    
    means = [[], []]
    stds = [[], []]
    for i in range(len(results)):
        for j in range(len(results[0])):
            arr = np.array(results[i][j])
            means[i].append(arr.mean())
            stds[i].append(arr.std())

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        mean = means[i]
        std = stds[i]
        axs[i].errorbar([j for j in range(len(mean))], mean, yerr=std)
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(mean)))
        axs[i].set_xticklabels(range(len(mean)))
        axs[i].axhline(y=mean[0], color="red", linestyle="--", linewidth=1, label="0 mean")

    plt.tight_layout()
    plt.show()

def box_plot(dirpath):
    labels, results = get_data(dirpath)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        data = results[i]
        axs[i].boxplot(data, whis=50)
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

def hist_plot(dirpath): # removing the top 80%
    labels, results = get_data(dirpath)
    
    n = len(results[0])
    size = 5
    fig, axs = plt.subplots(math.ceil(n / size)-1, min(n, size), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(1, n):
        x, y = (i-1) % size, (i-1) // size
        data = np.array(results[0][i])
        top = np.percentile(data, 20)
        axs[y][x].hist(data[data < top], bins=50)
        axs[y][x].grid(True)
        axs[y][x].set_title(i)
        axs[y][x].set_xlabel("Accuracy")
        axs[y][x].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def dist_plot(dirpath):
    labels, results = get_data(dirpath)
    
    n = len(results[0])
    fig, axs = plt.subplots(math.ceil(n / 6), min(n, 6), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(n):
        x, y = i % 6, i // 6
        data = np.array(results[0][i])
        axs[y][x].hist(data, bins=100)
        axs[y][x].set_title(i)

    plt.tight_layout()
    plt.show()
    

def bit_pos_err(dirpath):
    df = pd.read_csv(dirpath, names=["index", "same", "sim", "error"])
    df = df.drop(columns=["same"])
    df["sim"] = df["sim"].astype(float)
    df["error"] = df["error"].str.replace("max err - ", "").astype(float)
    n = df["index"].value_counts().mean()
    df = df.groupby("index").mean()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(df.index, df["sim"], color="blue", label="Similarity")
    ax1.set_xlabel("Bit position")
    ax1.set_ylabel("Similarity", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(df.index, df["error"], color="red", label="Max Error")
    ax2.set_xlabel("Bit position")
    ax2.set_ylabel("Max Error", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    plt.title(f"Error and Cosine similarity over bit position ({int(n)} tests)")
    plt.legend()
    plt.show()

dirpath = "../results/faults_output/"
hist_plot(dirpath)

# bit_pos_err("../results/bit_pos_test.txt")