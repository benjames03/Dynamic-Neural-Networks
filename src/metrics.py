import torch
import os
import pandas as pd

# Function to calculate precision for each class
def precision(conf_matrix):
    with torch.no_grad():
        true_positives = torch.diag(conf_matrix)
        predicted_positives = conf_matrix.sum(dim=0)
        precisions = true_positives / (predicted_positives + 1e-8)  # Adding epsilon to prevent division by zero
    return precisions

# Function to calculate recall for each class
def recall(conf_matrix):
    with torch.no_grad():
        true_positives = torch.diag(conf_matrix)
        actual_positives = conf_matrix.sum(dim=1)
        recalls = true_positives / (actual_positives + 1e-8)  # Adding epsilon to prevent division by zero
    return recalls

# Function to calculate F1 score for each class
def f1_score(precisions, recalls):
    with torch.no_grad():
        f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)  # Adding epsilon to prevent division by zero
    return f1s

def analyse():
    model = None # load model

    precisions = precision(model.conf_matrix)
    recalls = recall(model.conf_matrix)
    f1s = f1_score(precisions, recalls)

    print(model.conf_matrix)
    print("Precision per class:", precisions)
    print("Recall per class:", recalls)
    print("F1 score per class:", f1s)

def summarise_fault_tests():
    dirpath = "../results/faults/"
    base_fp, test_fps = "0.txt", ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt", "6.txt", "7.txt", "8.txt"]

    df = pd.read_csv(dirpath + base_fp, names=["accuracy", "margin"])
    base_acc, base_mar = df["accuracy"].mean(), df["margin"].mean()
    results = [f"{base_fp[:-4]} faults ({df.shape[0]} tests): {100*base_acc:.2f}%, {base_mar:.3g}\n"]
    for filepath in test_fps:
        df = pd.read_csv(dirpath + filepath, names=["accuracy", "margin"])
        results.append(f"{filepath[:-4]} faults ({df.shape[0]} tests): {100*(df['accuracy'].mean()-base_acc):+.2f}%, {df['margin'].mean()-base_mar:+.3g}\n")
    with open(dirpath + "summary.txt", "w") as file:
        for result in results:
            file.write(result)

summarise_fault_tests()