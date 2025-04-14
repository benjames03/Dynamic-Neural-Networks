import torch
import os
import pandas as pd
import numpy as np
import lenet
import resnet
from train import get_data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def precision(conf_matrix):
    with torch.no_grad():
        conf_matrix = torch.tensor(conf_matrix, dtype=torch.float32)
        true_positives = torch.diag(conf_matrix)
        predicted_positives = conf_matrix.sum(dim=0)
        precisions = true_positives / (predicted_positives + 1e-8)
    return precisions

def recall(conf_matrix):
    with torch.no_grad():
        conf_matrix = torch.tensor(conf_matrix, dtype=torch.float32)
        true_positives = torch.diag(conf_matrix)
        actual_positives = conf_matrix.sum(dim=1)
        recalls = true_positives / (actual_positives + 1e-8)
    return recalls

def f1_score(precisions, recalls):
    with torch.no_grad():
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    return f1s


def conf_matrix(model):
    _, _, dataloader = get_data(batch_size=32)
    model.eval()

    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    return confusion_matrix(y_true, y_pred)

def analyse():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = resnet.ResNet20().to(device)
    model.load_state_dict(torch.load("../models/resnet20.pth", weights_only=True))

    cm = np.array(conf_matrix(model))
    
    precisions = precision(cm)
    recalls = recall(cm)
    f1s = f1_score(precisions, recalls)

    print(cm)
    print("Precision per class:", precisions)
    print("Recall per class:", recalls)
    print("F1 score per class:", f1s)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    # plt.title("Confusion Matrix")
    plt.savefig("../results/graphs/resnet20_conf.pdf", format="pdf", bbox_inches="tight")
    plt.show()

def summarise_fault_tests(fileout, dirpath):
    files = sorted(os.listdir(dirpath), key=lambda fn: int(fn[:-4]))
    base_fn, test_fns = files[0], files[1:]

    df = pd.read_csv(dirpath + base_fn, names=["accuracy", "margin", "faults"])
    df["margin"] = df["margin"].astype(float)
    df = df.dropna()
    base_acc, base_mar = df["accuracy"].mean(), df["margin"].mean()
    results = [f"{base_fn[:-4]} faults ({df.shape[0]} tests): {100*base_acc:.2f}%, {base_mar:.3g}\n"]
    for test_fn in test_fns:
        df = pd.read_csv(dirpath + test_fn, names=["accuracy", "margin", "faults"])
        df["margin"] = df["margin"].astype(float)
        df = df.dropna()
        # df = df[df["margin"] != 0].dropna()
        results.append(f"{test_fn[:-4]} faults ({df.shape[0]} tests): {100*(df['accuracy'].mean()-base_acc):+.2f}%, {df['margin'].mean()-base_mar:+.3g}\n")
    with open(fileout, "w") as file:
        for result in results:
            file.write(result)

# fileout = "../results/0/summaries/output_fault.txt"
# dirpath = "../results/0/faults_output/"
# summarise_fault_tests(fileout, dirpath)
# analyse()