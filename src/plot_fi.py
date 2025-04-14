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
        df = pd.read_csv(dirpath + filepath)
        df.columns = ["accuracy", "margin"] if df.shape[1] == 2 else ["accuracy", "margin", "faults"]
        df["margin"] = df["margin"].astype(float)
        # df = df.dropna()
        df = df[df["margin"] != 0].dropna()
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
        axs[i].boxplot(data, whis=500, patch_artist=True)
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(data)))
        axs[i].set_xticklabels(range(len(data)))
        axs[i].axhline(y=data[0].mean(), color="green", linestyle="-", linewidth=1, label="0 fault")

    plt.tight_layout()
    # plt.savefig("../results/graphs/fault_bit.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def violin_plot(dirpath):
    labels, results = get_data(dirpath)

    fig, axs = plt.subplots(1, len(results), figsize=(12, 4))
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(len(results)):
        data = results[i]
        sns.violinplot(ax=axs[i], data=data, inner="box", bw_method=0.4)
        axs[i].set_title(labels[i] + " over #faults")
        axs[i].set_xlabel("Number of Faults")
        axs[i].set_ylabel(labels[i])
        axs[i].set_xticks(range(len(data)))
        axs[i].set_xticklabels(range(len(data)))
        axs[i].axhline(y=data[0].mean(), color="blue", linestyle="--", linewidth=1, label="0 fault")

    plt.tight_layout()
    # plt.savefig("../results/graphs/fault_bit.pdf", format="pdf", bbox_inches="tight")
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
    # plt.savefig("../results/graphs/out_1_strip.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def hist_plot(dirpath): # removing the top 80%
    labels, results = get_data(dirpath)
    
    p = 0 # 0 - acc, 1 - margin
    size = 2
    n = len(results[p]) - 1
    base_mean = np.array(results[p][0]).mean()

    fig, axs = plt.subplots(math.ceil(n / size), min(n, size), figsize=(8, 10), squeeze=False)
    plt.gcf().canvas.manager.set_window_title("Fault injection results")
    for i in range(1, n+1):
        x, y = (i-1) % size, (i-1) // size
        data = np.array(results[p][i])
        top = np.percentile(data, 20)
        # axs[y][x].hist(data, bins=50)
        axs[y][x].hist(data[data < top], bins=50)
        axs[y][x].grid(True)
        axs[y][x].set_title(f"{i} Fault" + ("" if i == 1 else "s"))
        axs[y][x].set_xlabel("Accuracy")
        axs[y][x].set_ylabel("Frequency")
        axs[y][x].axvline(x=base_mean, color="red", linestyle="-", linewidth=1, label="0 fault")

    plt.tight_layout(h_pad=2)
    plt.savefig("../results/graphs/out_1_hist.pdf", format="pdf", bbox_inches="tight")
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

    axs[0].set_title(f"Cosine Similarity over bit position")
    axs[0].plot(dfs[0].index, dfs[0]["sim"], color="blue", label="Set 0", marker="o", ms=3)
    axs[0].plot(dfs[1].index, dfs[1]["sim"], color="red", label="Set 1", marker="o", ms=3)
    axs[0].set_xlabel("Faulty Bit position")
    axs[0].set_ylabel("Cosine Similarity")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title(f"Maximum Error over bit position")
    axs[1].plot(dfs[0].index, dfs[0]["error"], color="blue", label="Set 0", marker="o", ms=3)
    axs[1].plot(dfs[1].index, dfs[1]["error"], color="red", label="Set 1", marker="o", ms=3)
    axs[1].set_xlabel("Faulty Bit position")
    axs[1].set_ylabel("Max. Error")
    axs[1].legend()
    axs[1].grid()

    # axs[0].set_yscale("symlog", linthresh=1e-0)
    axs[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig("../results/graphs/fault_bit.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__": # python3 plot_f1.py 0 output box
    if len(sys.argv) != 4:
        print("Incorrect number of arguments")
        sys.exit()
    
    set = sys.argv[1]
    if set != "0" and set != "1":
        print("Incorrect set [0, 1]")
        sys.exit()
    
    dirpath = "../results/" + set + "/faults_"
    if (file := sys.argv[2]) == "output" or file == "kernel" or file == "ubiq":
        dirpath += file + "/"
    else:
        print(f"Incorrect directory path: {file} [output, kernel, ubiq]")
        sys.exit()

    if (plot := sys.argv[3]) == "basic":
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
    