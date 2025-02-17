import time
import random
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import lenet

DATASET_PATH = "../datasets/cifar10"
MODEL_PATH = "../models/lenet.pth"
RESULTS_PATH = "../results/faults/"

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

def eval_submodel(args):
    loader, faults = args
    model = lenet.SimLeNet().to("cpu")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.inject_faults(faults)

    model.eval()
    total_acc, total_margin = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to("cpu"), y.to("cpu")

            pred = model(X)
            total_acc += (pred.argmax(1) == y).sum().item()
            if not torch.isnan(pred).any():
                probs = F.softmax(pred, dim=1)
                vals, _ = torch.topk(probs, 2, dim=1)
                total_margin += torch.sum(vals[:, 0] - vals[:, 1]).item()
        
    total_acc /= len(loader.dataset)
    total_margin /= len(loader.dataset)
    return [total_acc, total_margin]

# use mp.set_start_method("spawn")
def full_inference(loaders, num_faults):
    kernels, multipliers, bits = 16, 64, 32
    faults = [(random.randint(0, kernels-1), random.randint(0, multipliers-1), random.randint(0, bits-1)) for _ in range(num_faults)]
    with mp.Pool(processes=len(loaders)) as pool:
        results = pool.map(eval_submodel, [(loader, faults) for loader in loaders])
        mean_acc = sum(result[0] for result in results) / len(results)
        mean_mar = sum(result[1] for result in results) / len(results)
    return (mean_acc, mean_mar)

def append_record(num_faults, accuracy, margin):
    with open(RESULTS_PATH + str(num_faults) + ".txt", "a") as file:
        file.write(f"{accuracy}, {margin}\n")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    loaders = get_data_mp(batch_size=50, num_loaders=4)
    num_faults = 3
    num_tests = 1000
    for i in range(num_tests):
        start = time.time()
        (accuracy, margin) = full_inference(loaders, num_faults)
        # append_record(num_faults, accuracy, margin)
        stop = time.time()
        print(f"\r{i+1}/{num_tests} tests ({stop-start:.2f}s)", end="")
    print()