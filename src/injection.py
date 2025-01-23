import time
import torch
from torch.utils.data import DataLoader, Subset
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
    total_acc = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to("cpu"), y.to("cpu")

            pred = model(X)
            total_acc += (pred.argmax(1) == y).sum().item()

    total_acc /= len(loader.dataset)
    return total_acc

# use mp.set_start_method("spawn")
def full_inference(loaders, faults):
    with mp.Pool(processes=len(loaders)) as pool:
        results = pool.map(eval_submodel, [(loader, faults) for loader in loaders])
        mean_acc = sum(results) / len(results)
    return mean_acc

def append_record(faults, accuracy, time):
    with open(RESULTS_PATH + str(faults) + ".txt", "a") as file:
        file.write(f"{accuracy}, {time:.2f}\n")

if __name__ == "__main__":
    mp.set_start_method("spawn")
    loaders = get_data_mp(batch_size=25, num_loaders=4)
    faults = 1
    tests = 20
    for i in range(tests):
        start = time.time()
        accuracy = full_inference(loaders, faults)
        stop = time.time()
        append_record(faults, accuracy, stop-start)
        print(f"\r{i+1}/{tests}", end="")