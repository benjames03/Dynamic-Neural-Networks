import os
import sys
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
        df = pd.read_csv(dirpath + filepath, names=["accuracy", "margin", "faults"])
        df["margin"] = df["margin"].astype(float)
        df = df.dropna()
        data[0].append(df["accuracy"].to_numpy())
        data[1].append(df["margin"].to_numpy())

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
    addends = ["0.txt", "1.txt"]
    dfs = [None, None]

    fig, axs = plt.subplots(1, len(addends), figsize=(15, 5))

    for i in range(len(addends)):
        dfs[i] = pd.read_csv(dirpath + addends[i], names=["index", "same", "sim", "error"])
        dfs[i] = dfs[i].drop(columns=["same"])
        dfs[i]["sim"] = dfs[i]["sim"].astype(float)
        dfs[i]["error"] = dfs[i]["error"].str.replace("max err - ", "").astype(float)
        n = dfs[i]["index"].value_counts().mean()
        dfs[i] = dfs[i].groupby("index").mean()

        axs[i].plot(dfs[i].index, dfs[i]["sim"], color="blue", label="Similarity")
        axs[i].set_xlabel("Bit position")
        axs[i].set_ylabel("Similarity", color="blue")

        ax2 = axs[i].twinx()
        ax2.grid(True)
        ax2.plot(dfs[i].index, dfs[i]["error"], color="red", label="Max Error")
        ax2.set_xlabel("Bit position")
        ax2.set_ylabel("Max Error", color="red")

    axs[0].set_title(f"Error and Cosine similarity over bit position ({int(n)} tests, set 0)")
    axs[1].set_title(f"Error and Cosine similarity over bit position ({int(n)} tests, set 1)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect number of arguments")
        sys.exit()
    
    dirpath = "../results/faults_"
    if (file := sys.argv[1]) == "output" or file == "kernel" or file == "ubiq":
        dirpath += file + "/"
    else:
        print(f"Incorrect directory path: {file} [output, kernel, ubiq]")
        sys.exit()

    if (plot := sys.argv[2]) == "basic":
        basic_plot(dirpath)
    elif plot == "box":
        box_plot(dirpath)
    elif plot == "violin":
        violin_plot(dirpath)
    elif plot == "strip":
        strip_plot(dirpath)
    elif plot == "hist":
        hist_plot(dirpath)
    elif plot == "dist":
        dist_plot(dirpath)
    elif plot == "bit":
        bit_pos_err("../results/bit_pos_test_")
    else:
        print(f"Incorrect plot type: {plot} [basic, box, violin, strip, hist, dist, bit]")
        sys.exit()
    