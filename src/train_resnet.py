import time
import torch
import torchvision
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip
import resnet

DATASET_PATH = "../datasets/cifar10"
MODELS_PATH = "../models/"

def get_data(batch_size):
    train_transform = Compose([
        Pad(4),
        RandomCrop(32),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    test_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])

    # Download training data from open datasets.
    training_data = datasets.CIFAR10(
        root=DATASET_PATH,
        train=True,
        download=True,
        transform=train_transform,
    )

    # Download test data from open datasets.
    test_data = datasets.CIFAR10(
        root=DATASET_PATH,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader

def train(dataloader, model, loss_function, optimizer):
    model.train()
    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to("cpu"), y.to("cpu")
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    print(f"Train Accuracy: {(100*train_acc):>0.1f}%, Train loss: {train_loss:>8f}")

def test(dataloader, model, loss_function):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to("cpu"), y.to("cpu")

            pred = model(X)
            loss = loss_function(pred, y)

            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)
    print(f"Test Accuracy: {(100*test_acc):>0.1f}%, Test loss: {test_loss:>8f}")
    return test_acc, test_loss

def save_model(filename):
    filepath = MODELS_PATH + filename
    torch.save(model.state_dict(), filepath)
    print(f"Saved PyTorch Model State to {filepath}")

train_dataloader, test_dataloader = get_data(batch_size=32)
model = resnet.ResNet18()
loss_function = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000125)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

epochs = 16
stats = np.zeros((3, epochs)) # accuracy, loss, time
for epoch in range(epochs):
    start = time.time()
    print(f"----------------- EPOCH {epoch+1} --------------")

    train(train_dataloader, model, loss_function, optimizer)
    stats[0][epoch], stats[1][epoch] = test(test_dataloader, model, loss_function)
    scheduler.step()

    end = time.time()
    stats[2][epoch] = end - start
    print(f"{stats[2][epoch]:.2f}s")

total_time = np.sum(stats[2])
print(f"Finished in {total_time//60:.0f}mins{total_time%60:.0f}s")
save_model("resnet18.pth")