import time
import torch
from torchinfo import summary
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

import lenet
import resnet
import big_little

DATASET_PATH = "../../datasets/cifar10"
MODELS_PATH = "../models/"

def get_data(batch_size):
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_data = datasets.CIFAR10(
        root=DATASET_PATH,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(test_data, batch_size=batch_size, shuffle=True)

def load_lenet():
    model = lenet.LeNet().to("cpu")
    model.load_state_dict(torch.load(f"../models/lenet.pth", weights_only=True))
    return model

def load_resnet9():
    model = resnet.ResNet9().to("cpu")
    model.load_state_dict(torch.load(f"../models/resnet9.pth", weights_only=True))
    return model

def load_resnet18():
    model = resnet.ResNet18().to("cpu")
    model.load_state_dict(torch.load(f"../models/resnet18.pth", weights_only=True))
    return model

def load_resnet20():
    model = resnet.ResNet20().to("cpu")
    model.load_state_dict(torch.load(f"../models/resnet20.pth", weights_only=True))
    return model

def test(dataloader, model, loss_fn, threshold):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")

            pred = model(X, threshold=threshold, labels=y)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    return correct, test_loss

start = time.time()

loader = get_data(batch_size=32)

lenet = load_lenet()
resnets = [load_resnet9(), load_resnet20()]
loss_fn = nn.CrossEntropyLoss()

n = 21
data = np.zeros(2, dtype=[
    ("thresholds", "f4", (n,)),  
    ("accuracies", "f4", (n,)),  
    ("losses", "f4", (n,)),      
    ("cpu_time", "f4", (n,)), 
    ("operations", "f4", (n,))
])
data[0]["thresholds"] = np.linspace(0, 1, n)
data[1]["thresholds"] = np.linspace(0, 1, n)

for i, resnet in enumerate(resnets):
    print("---------------------------")
    model = big_little.Model(lenet, resnet).to("cpu")
    for j, threshold in enumerate(data[i]["thresholds"]):
        go = time.time()

        data[i]["accuracies"][j], data[i]["losses"][j] = test(loader, model, loss_fn, threshold)

        stop = time.time()
        data[i]["cpu_time"][j] = stop - go
        data[i]["operations"][j] = model.resnet_calls

        print(f"Threshold {threshold:.2f} finished in {stop - go:.2f} seconds")
        print(f"Resnet called {model.resnet_calls}/{model.lenet_calls} times ({100*model.resnet_calls/model.lenet_calls:.1f}%)")
        model.lenet_calls, model.resnet_calls = 0, 0

    lenet_ops = summary(lenet, input_size=(1, 3, 32, 32), verbose=0).total_mult_adds
    resnet_ops = summary(resnet, input_size=(1, 3, 32, 32), verbose=0).total_mult_adds

    data[i]["operations"] = (data[i]["operations"] * resnet_ops + 10000 * lenet_ops) * 1e-12

np.save("../results/conf_matrix.npy", model.conf_matrix)
np.save("../results/exp_data.npy", data)

end = time.time()
print(f"Finished in {end-start:.2f} seconds")
