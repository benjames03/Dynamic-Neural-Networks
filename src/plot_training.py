import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot(path, version, files):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    skip = 0
    if files[0] == "all":
        files = os.listdir(path)
    elif files[0] == "all10":
        files = [file for file in os.listdir(path) if file.endswith("10.txt")]
    elif files[0] == "all20":
        files = [file for file in os.listdir(path) if file.endswith("20.txt")]
    elif files[0] == "all50":
        files = [file for file in os.listdir(path) if file.endswith("50.txt")]

    labels = []#["0", "0.0001", "0.00001", "GELU", "50"]
    add = "Weight decay"

    for i, file in enumerate(files):
        df = pd.read_csv(path + file, sep=" ")
        # print(df)
        df = df.T.reset_index()
        df.columns = ["t_acc", "t_loss", "v_acc", "v_loss"]
        df = df.astype(float)
        label = file
        if labels != []:
            label = labels[i]
        axs[0].plot(df.index[skip:], df[version + "_acc"][skip:], label=f"{label}", marker="o", markersize=3)
        axs[1].plot(df.index[skip:], df[version + "_loss"][skip:], label=f"{label}", marker="o", markersize=3)

    label = "Training" if version == "t" else "Validation"
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title(f"{label} Accuracy over Epochs ({add})")
    axs[0].legend()
    # axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[0].grid(True)

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title(f"{label} Loss over Epochs ({add})")
    axs[1].legend()
    # axs[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("../results/graphs/lenet_decay.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def plot_final(path, files):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    skip = 0

    for file in files:
        df = pd.read_csv(path + file, sep=" ")
        df = df.T.reset_index()
        df.columns = ["t_acc", "t_loss", "v_acc", "v_loss"]
        df = df.astype(float)
        axs[0].plot(df.index[skip:], df["t_acc"][skip:], label="25 epochs (training)", marker="o", markersize=3)
        axs[0].plot(df.index[skip:], df["v_acc"][skip:], label="25 epochs (validation)", marker="o", markersize=3)
        axs[1].plot(df.index[skip:], df["t_loss"][skip:], label="50 epochs (training)", marker="o", markersize=3)
        axs[1].plot(df.index[skip:], df["v_loss"][skip:], label="50 epochs (validation)", marker="o", markersize=3)

    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_title("Accuracy over Epochs")
    axs[0].legend()
    # axs[0].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[0].grid(True)

    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Loss over Epochs")
    axs[1].legend()
    # axs[1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("../results/graphs/resnet20_final.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__": # python3 plot_training.py ../results/training/lenet v all
    if len(sys.argv) < 3:
        print("Incorrect number of arguments")
        sys.exit()
    path = sys.argv[1]
    version = sys.argv[2]
    files = sys.argv[3:]
    if version == "f":
        plot_final(path, files)
    else:
        plot(path, version, files)

"""
epochs
python3 plot_training.py ../results/training/lenet/ v base_10.txt base_15.txt base_20.txt base_25.txt base_50.txt

batch_size
python3 plot_training.py ../results/training/lenet/ v batch1_20.txt batch16_20.txt base_20.txt batch64_20.txt batch128_20.txt

learning_rate
python3 plot_training.py ../results/training/lenet/ v base_20.txt lr002_20.txt lr0015_20.txt

activation
python3 plot_training.py ../results/training/lenet/ v base_10.txt sig_10.txt tanh_10.txt gelu_10.txt

weight_decay
python3 plot_training.py ../results/training/lenet/ v base_20.txt relu_d0001_20.txt relu_d00001_20.txt

lenet: Test Accuracy: 71.4%, Test loss: 0.820399 17mins59s
resnet9 10: Test Accuracy: 84.2%, Test loss: 0.462832 17mins4s
resnet9 15: Test Accuracy: 86.8%, Test loss: 0.381852 27mins0s
resnet20 15: Test Accuracy: 87.3%, Test loss: 0.372168 18mins28s
resnet20 20: Test Accuracy: 88.7%, Test loss: 0.342486 24mins30s
resnet20 20 m7: Test Accuracy: 88.7%, Test loss: 0.340632 25mins51s
resnet20 25 m7 emin5e4: Test Accuracy: 89.0%, Test loss: 0.343035 30mins49s
resnet20 25 m7 emin5e4 d5e3: Test Accuracy: 86.8%, Test loss: 0.400280 30mins50s
resnet20 50 m7 emin5e4: Test Accuracy: 90.5%, Test loss: 0.295263 62mins8s
"""