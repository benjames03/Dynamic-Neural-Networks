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
RESULTS_PATH = "../results/0/faults_kernel/"

float_type = torch.float32 # torch.float32 or torch.float16
torch.set_default_dtype(float_type)

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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    loader, faults, set, method = args
    if all(fault[1] > 29 for fault in faults) or method == "ker":
        model = lenet.LeNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        # if method == "ker":
        #     model.inject_faults(faults, set)
    else:
        model = lenet.SimLeNet().to(device)
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
        model.inject_faults(faults, set, method)
        model.to(float_type)

    model.eval()
    total_acc, total_margin = 0, 0
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)

            pred = model(X)
            total_acc += (pred.argmax(1) == y).sum().item()
            if not torch.isnan(pred).any():
                probs = F.softmax(pred, dim=1)
                vals, _ = torch.topk(probs, 2, dim=1)
                total_margin += torch.sum(vals[:, 0] - vals[:, 1]).item()
            else:
                print("nan")
            print(f"\r{i+1}/{len(loader)}", end="")
        
    total_acc /= len(loader.dataset)
    total_margin /= len(loader.dataset)
    return [total_acc, total_margin]

# use mp.set_start_method("spawn")
def full_inference(loaders, num_faults, set=1, method="out"):
    macs, multipliers, bits = 16, 64, 32
    faults = [(random.randrange(macs), random.randrange(multipliers), random.randrange(bits)) for _ in range(num_faults)]
    with mp.Pool(processes=len(loaders)) as pool:
        results = pool.map(eval_submodel, [(loader, faults, set, method) for loader in loaders])
        mean_acc = sum(result[0] for result in results) / len(results)
        mean_mar = sum(result[1] for result in results) / len(results)
    return (mean_acc, mean_mar, faults)

def append_record(num_faults, accuracy, margin, faults):
    with open(RESULTS_PATH + str(num_faults) + ".txt", "a") as file:
        file.write(f'{accuracy}, {margin}, {str(faults).replace("), (", ")-(").replace(",", "")}\n')

if __name__ == "__main__":
    mp.set_start_method("spawn")
    loaders = get_data_mp(batch_size=500, num_loaders=2)
    num_faults = 10
    num_tests = 100
    set = 0
    method = "ker" # "ker" or "out"
    for i in range(num_tests):
        start = time.time()
        (accuracy, margin, faults) = full_inference(loaders, num_faults, set, method)
        # append_record(num_faults, accuracy, margin, faults)
        print("\r", set, accuracy, margin, faults)
        stop = time.time()
        print(f"\r{i+1}/{num_tests} tests ({stop-start:.2f}s)")

"""
1 fault, 2 processes
batch_size (est time):
    50 ~ 240s
   100 ~ 141s
   200 ~ 93s
   500 ~ 66s
  1000 ~ 74s
"""