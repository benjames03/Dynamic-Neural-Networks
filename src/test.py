import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
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

def load_lenet(simulating=False):
    model = lenet.LeNet(simulating).to("cpu")
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

        test_loss /= len(loader)
        total_acc /= len(loader.dataset)
        stop = time.time()
        if state: print(f"({stop-go:.2f}s)")
 
    torch.save(accuracies, "../results/conv_comp.pt")

def eval_conv_layer():
    start = time.time()
    input_cube = torch.rand((32, 3, 32, 32))
    simulated = conv.DirectConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark.weight = simulated.weight
    # benchmark.bias = simulated.bias
    with torch.no_grad():
        benchmark.bias.zero_()
        simulated.bias.zero_()
    a = simulated(input_cube)
    b = benchmark(input_cube)
    print(torch.equal(a, b),
            F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "mean dif -", torch.abs(a - b).mean().item())
    print(f"{time.time()-start:.2f}s")

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
                    C = conv.DirectConv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=padding, dilation=1)
                    a = C.conv2d(input_cube)
                    b = F.conv2d(input_cube, C.weight, C.bias, stride=stride, padding=padding)
                    print(f"pad={padding}, stride={stride} ->", torch.equal(a, b),
                        F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
                        "mean dif -", torch.abs(a - b).mean())
                    
eval_model()