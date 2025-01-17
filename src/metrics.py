import torch

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

model = None # load model

precisions = precision(model.conf_matrix)
recalls = recall(model.conf_matrix)
f1s = f1_score(precisions, recalls)

print(model.conf_matrix)
print("Precision per class:", precisions)
print("Recall per class:", recalls)
print("F1 score per class:", f1s)