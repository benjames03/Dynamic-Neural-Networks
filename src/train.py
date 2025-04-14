import sys
import time
import numpy as np
import torch
from torch import nn, no_grad, float, save, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize, Pad, RandomCrop, RandomHorizontalFlip
import lenet, resnet

DATASET_PATH = "../datasets/cifar10"
MODELS_PATH = "../models/"

def get_data(batch_size, split=0.2):
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

    full_training_data = datasets.CIFAR10(
        root=DATASET_PATH,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = datasets.CIFAR10(
        root=DATASET_PATH,
        train=False,
        download=True,
        transform=test_transform,
    )

    total_train = len(full_training_data)
    val_size = int(total_train * split)
    train_size = total_train - val_size

    train_data, val_data = random_split(full_training_data, [train_size, val_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def train(dataloader, model, loss_function, optimizer):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.train()
    train_loss, train_acc = 0, 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        pred = model(X)
        loss = loss_function(pred, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(float).sum().item()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader.dataset)
    print(f"Train Accuracy: {(100*train_acc):>0.1f}%, Train loss: {train_loss:>8f}")
    return [train_acc, train_loss]

def test(dataloader, model, loss_function):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()
    test_loss, test_acc = 0, 0
    with no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = loss_function(pred, y)

            test_loss += loss.item()
            test_acc += (pred.argmax(1) == y).type(float).sum().item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader.dataset)
    print(f"Test Accuracy: {(100*test_acc):>0.1f}%, Test loss: {test_loss:>8f}")
    return [test_acc, test_loss]

def save_model(model, filename):
    filepath = MODELS_PATH + filename + "_v4.pth"
    save(model.state_dict(), filepath)
    print(f"Saved PyTorch Model State to {filepath}")

def save_results(filename, results):
    version = "w10_c20_m7_10_1_01"
    filepath = "../results/training/" + filename + "/" + version + ".txt"
    np.savetxt(filepath, results)

def process(model, model_filename, epochs):
    train_dataloader, val_dataloader, test_dataloader = get_data(batch_size=32)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.7, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 0.0005)

    start = time.time()
    results = np.zeros((4, epochs)) # accuracy, loss
    for epoch in range(epochs):
        print(f"----------------- EPOCH {epoch+1} --------------")

        results[:2, epoch] = train(train_dataloader, model, loss_function, optimizer)
        results[2:, epoch] = test(val_dataloader, model, loss_function)
        scheduler.step()

    test(test_dataloader, model, loss_function)

    end = time.time()
    total_time = end - start
    print(f"Finished in {total_time//60:.0f}mins{total_time%60:.0f}s")

    save_model(model, model_filename)
    save_results(model_filename, results)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Incorrect number of arguments")
        sys.exit()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_name = sys.argv[1]
    epochs = int(sys.argv[2])
    model = None
    if model_name == "lenet":
        model = lenet.LeNet().to(device)
    elif model_name == "resnet9":
        model = resnet.ResNet9().to(device)
    elif model_name == "resnet18":
        model = resnet.ResNet18().to(device)
    elif model_name == "resnet20":
        model = resnet.ResNet20().to(device)
    else:
        print("Incorrect model name, options: lenet, resnet9, resnet18, resnet20")
        sys.exit()
    process(model, model_name, epochs)
    