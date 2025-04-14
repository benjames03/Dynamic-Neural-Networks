import time
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import lenet, conv

DATASET_PATH = "../datasets/cifar10"
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    if simulating:
        model = lenet.SimLeNet().to(device)
    else:
        model = lenet.LeNet().to(device)
    model.load_state_dict(torch.load(f"../models/lenet.pth", weights_only=True))
    return model

def eval_model(checkpoint=500):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
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
                X, y = X.to(device), y.to(device)

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = lenet.SimLeNet().to(device)
    model.load_state_dict(torch.load(f"../models/lenet.pth", weights_only=True))
    model.eval()

    total_acc = 0
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

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
    input_cube = torch.rand((50, 3, 32, 32))
    simulated = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    benchmark.weight = simulated.weight
    benchmark.bias = simulated.bias
    a = simulated(input_cube)
    b = benchmark(input_cube)
    print(torch.equal(a, b),
            F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "mean dif -", torch.abs(a - b).mean().item())
    
def eval_bit_flips():
    bit = 30
    num_f = torch.tensor(0x02020202, dtype=torch.float32)
    num_i = num_f.view(torch.int32)
    mask1 = torch.tensor(2**bit, dtype=torch.int32)
    mask0 = ~mask1
    print("Bit    =", bit)
    print("1-Mask =", format(mask1 & 0xffffffff, "032b"))
    print("0-Mask =", format(mask0 & 0xffffffff, "032b"))

    set1 = num_i | mask1
    set0 = num_i & mask0
    print("Number =", format(num_i & 0xffffffff, "032b"), num_f)
    print("1-set  =", format(set1 & 0xffffffff, "032b"), set1.view(torch.float32))
    print("0-set  =", format(set0 & 0xffffffff, "032b"), set0.view(torch.float32))
    
def eval_faulty_conv_layer():
    num_faults = 1
    macs, multipliers, bits = 16, 64, 32
    # choose random mac, random mult (within range), and set bit 30
    faults = [(random.randrange(macs), random.randrange(3), 30) for _ in range(num_faults)]
    print(faults)
    
    input_cube = torch.rand((50, 3, 32, 32))
    benchmark = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    simulated_free = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    simulated_faulty_0 = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    simulated_faulty_1 = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)

    simulated_faulty_1.weight = benchmark.weight
    simulated_faulty_1.bias = benchmark.bias
    simulated_faulty_0.weight = benchmark.weight
    simulated_faulty_0.bias = benchmark.bias
    simulated_free.weight = benchmark.weight
    simulated_free.bias = benchmark.bias
    simulated_faulty_1.inject_faults(faults, 1, "out")
    simulated_faulty_0.inject_faults(faults, 0, "out")

    a = benchmark(input_cube)
    b = simulated_free(input_cube)
    c = simulated_faulty_1(input_cube)
    d = simulated_faulty_0(input_cube)

    print("Simulated:", torch.equal(a, b),
            F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
            "max error -", torch.max(a - b).mean().item())
    print("Faulty  1:", torch.equal(a, c),
            F.cosine_similarity(a.flatten(), c.flatten(), dim=0).item(),
            "max error -", torch.max(a - c).mean().item())
    print("Faulty  0:", torch.equal(a, d),
            F.cosine_similarity(a.flatten(), d.flatten(), dim=0).item(),
            "max error -", torch.max(a - d).mean().item())
    # print(F.mse_loss(c, a))
    # print((a == c).float().mean())
    # print(torch.abs(a - c).mean())
    # print(torch.norm(a - c, p=2))

def record_bit_errors(set=1):
    output_file = f"../results/bit_pos_test_{set}.txt"
    results = []
    macs, multipliers, bits = random.randrange(16), random.randrange(3), 32
    num_faults = 1
    input_cube = torch.rand((32, 3, 32, 32))
    benchmark = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
    for i in range(bits):
        faults = [(macs, multipliers, i) for _ in range(num_faults)]
        
        simulated_faulty = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1)
        simulated_faulty.weight = benchmark.weight
        simulated_faulty.bias = benchmark.bias
        simulated_faulty.inject_faults(faults, 0, "out")

        a = benchmark(input_cube)
        c = simulated_faulty(input_cube)
        with open(output_file, "a") as file:
            file.write(f"{i}, {torch.equal(a, c)}, {F.cosine_similarity(a.flatten(), c.flatten(), dim=0).item()}, max err - {torch.max(a - c).mean().item()}\n")

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    loader = get_data(batch_size=1)
    model = load_lenet(True)
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            # print((pred.argmax(1) == y).sum().item())
            break

if __name__ == "__main__":
    start = time.time()

    eval_faulty_conv_layer()
    # record_bit_errors()
    # eval_bit_flips()

    end = time.time()
    print(f"({end-start:.2f}s)")