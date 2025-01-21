import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import lenet, conv

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

def get_data_mp(batch_size, num_loaders):
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

    dataset_size = len(test_data)
    chunk_size = dataset_size // num_loaders
    indices = list(range(dataset_size))

    loaders = [None] * num_loaders
    for i in range(num_loaders):
        start = i * chunk_size
        end = min(start + chunk_size, dataset_size)
        subset = Subset(test_data, indices[start:end])
        loaders[i] = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loaders

def load_lenet(simulating=False):
    if simulating:
        model = lenet.SimLeNet().to("cpu")
    else:
        model = lenet.LeNet().to("cpu")
    model.load_state_dict(torch.load(f"../models/lenet.pth", weights_only=True))
    return model

def eval_model(checkpoint=500):
    loader = get_data(batch_size=1)
    loss_fn = nn.CrossEntropyLoss()

    accuracies = torch.zeros((2, int(len(loader) / checkpoint)))

    for (s, state) in [(0, False), (1, True)]:
        go = time.time()
        if state: print("(Simulation)")

        model = load_lenet(state)
        model.eval()
        test_loss, total_acc = 0, 0
        with torch.no_grad():
            for i, (X, y) in enumerate(loader):
                X, y = X.to("cpu"), y.to("cpu")

                pred = model(X)

                test_loss += loss_fn(pred, y).item()
                total_acc += (pred.argmax(1) == y).sum().item() #type(torch.float)
                if state: print(f"\rBatch: {i+1}/{len(loader)}", end="")
                if (i+1) % checkpoint == 0: 
                    current_acc = total_acc / (i+1)
                    accuracies[s, int((i+1) / checkpoint) - 1] = current_acc
                    print(f"\rSample {i+1}: {100 * current_acc}%")
                    break

        test_loss /= len(loader)
        total_acc /= len(loader.dataset)
        stop = time.time()
        if state: print(f"({stop-go:.2f}s)")
 
    # torch.save(accuracies, "../results/conv_comp.pt")

def eval_submodel(loader):
    model = lenet.SimLeNet().to("cpu")
    model.load_state_dict(torch.load(f"../models/lenet.pth", weights_only=True))
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to("cpu"), y.to("cpu")

            pred = model(X)
            total_acc += (pred.argmax(1) == y).sum().item()

    total_acc /= len(loader.dataset)
    return total_acc

# use mp.set_start_method("spawn")
def eval_model_mp(num_loaders=4):
    loaders = get_data_mp(batch_size=25, num_loaders=num_loaders)

    print("Running", num_loaders, "threads")
    with mp.Pool(processes=num_loaders) as pool:
        results = pool.map(eval_submodel, loaders)
        mean_acc = sum(results) / len(results)
        return mean_acc

def eval_conv_layer():
    input_cube = torch.rand((32, 3, 32, 32))
    simulated = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark.weight = simulated.weight
    benchmark.bias = simulated.bias
    a = simulated(input_cube)
    b = benchmark(input_cube)
    print(torch.equal(a, b),
            F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "mean dif -", torch.abs(a - b).mean().item())

def eval_conv():
    paddings = [1]
    strides = [1]
    in_channels = [3]
    out_channels = [30]
    for in_channel in in_channels:
        for out_channel in out_channels:
            input_cube = torch.rand((1, in_channel, 32, 32))
            for padding in paddings:
                for stride in strides:
                    C = conv.SimConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=padding, dilation=1)
                    a = C.conv2d(input_cube)
                    b = F.conv2d(input_cube, C.weight, C.bias, stride=stride, padding=padding)
                    print(f"pad={padding}, stride={stride} ->", torch.equal(a, b),
                        F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                        "mean dif -", torch.abs(a - b).mean())
                    
def eval_linear_to_conv_layer():
    input = torch.rand((1, 208))

    linear = nn.Linear(208, 120)
    output1 = linear(input)

    input = input.view(1, 1, linear.weight.shape[1])
    weights = linear.weight
    weights = weights.view(weights.shape[0], 1, 1, weights.shape[1])
    conv2d = nn.Conv2d(in_channels=1, out_channels=weights.shape[0], kernel_size=(1, weights.shape[3]), stride=1, padding=0)
    conv2d.weight = nn.Parameter(weights)
    conv2d.bias = linear.bias
    output2 = conv2d(input)
    output2 = output2.view(output2.shape[1], output2.shape[0])

    print(output1.shape)
    print(output2.shape)

    print(torch.equal(output1, output2),
            F.cosine_similarity(output1.flatten(), output2.flatten(), dim=0).item(),
                        "mean dif -", torch.abs(output1 - output2).mean())

def eval_linear_to_conv_model():
    loader = get_data(batch_size=1)
    model = load_lenet(True)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to("cpu"), y.to("cpu")
            pred = model(X)
            # print((pred.argmax(1) == y).sum().item())
            break

if __name__ == "__main__":
    start = time.time()

    mp.set_start_method("spawn")
    accuracy = eval_model_mp(num_loaders=4)

    end = time.time()
    print(f"({end-start:.2f}s)")

"""
batch_size = 1:
4 threads: 1016.58s
8 threads: 1080.94s

batch_size = 25:
4 threads: 128.51s
8 threads: 350.96s
""" 