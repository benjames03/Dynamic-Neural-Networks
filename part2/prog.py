# modules
import numpy as np
import torch.nn as nn

# MAC classes
class MACArray:
    def __init__(self, size=16, multipliers=64):
        self.size = size
        self.cells = [MACCell(multipliers) for _ in range(size)]
        self.multipliers = multipliers
    
    def __str__(self):
        str = ""
        for i, cell in enumerate(self.cells):
            str += f"Cell {i:02} - {cell}\n"
        return str

    def broadcast(self, input_weights):
        if len(input_weights) == self.multipliers:
            for cell in self.cells:
                cell.mult(input_weights)

    def combine(self):
        return np.array([cell.accumulator for cell in self.cells], dtype=np.float64)

class MACCell:
    def __init__(self, multipliers=64):
        self.weights = np.zeros(multipliers)
        self.accumulator = 0

    def __str__(self):
        return np.array2string(self.weights[:10], separator=" ", threshold=0)

    def load(self, weights):
        self.weights = weights

    def mult(self, weights):
        self.accumulator = np.sum(self.weights * weights)

class DirectConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DirectConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def atomic(self, input_data, pos, depth):
        # broadcast the input data and multiply and accumulate for each cell
        mac_array.broadcast(input_data[pos][64*depth:64*(depth+1)])
        return mac_array.combine()

    def stripe(self, input_data, output_shape, depth, kpos):
        result = np.empty(output_shape)
        for oy in range(output_shape[1]):
            for ox in range(output_shape[0]):
                ix = ox + kpos[0]
                iy = oy + kpos[1]
                if ix < input_data.shape[0] and iy < input_data.shape[1]:
                    result[ox, oy] = atomic(input_data, (ix, iy), depth)
        return result

    def block(self, input_data, kernels, output_shape, depth):
        acc = np.zeros(output_shape)
        for y in range(kernels.shape[2]):
            for x in range(kernels.shape[1]):
                load_kernels(kernels, (x, y, depth))
                acc += stripe(input_data, output_shape, depth, (x, y))
        return acc

    def channel(self, input_data, kernels, output_shape):
        block_ops = int((input_data.shape[-1] + 63) / 64)
        blocks = np.zeros(output_shape, dtype=np.float64)
        for depth in range(block_ops):
            blocks += block(input_data, kernels, output_shape, depth)
        return blocks


def load_kernels(kernels, pos): # e.g. (x, y, z)
    # cache kernels into macs
    p = 64 * pos[-1]
    for i in range(mac_array.size):
        mac_array.cells[i].load(kernels[i][pos[:-1]][p:p+64]) # load kernel weights into each mac cell

def convolution(input_cube, kernels):
    stride = (1, 1) # x, y
    dil = (1, 1) # x, y
    pad = (0, 0, 0, 0) # left, right, top, bottom

    output_shape = (int((pad[0] + input_cube.shape[0] + pad[1] - (kernels.shape[1] - 1) * dil[0] - 1) / stride[0] + 1),
                    int((pad[2] + input_cube.shape[1] + pad[3] - (kernels.shape[2] - 1) * dil[1] - 1) / stride[1] + 1),
                    kernels.shape[0])

    blocks = channel(input_cube, kernels, output_shape)
    return blocks

def create_data(input_shape, kernel_shape):
    # set up input and kernels
    input_cube = np.ones(input_shape, dtype=np.float64)
    for i in range(input_cube.shape[0]):
        for j in range(input_cube.shape[1]):
            for k in range(input_cube.shape[2]):
                input_cube[i, j, k] = i + j * input_cube.shape[0]
    kernels = np.zeros(kernel_shape, dtype=np.float64)
    for i in range(kernels.shape[0]):
        kernels[i, :, :, :] = i
    
    return input_cube, kernels

mac_array = MACArray(size=16, multipliers=64)

input_cube, kernels = create_data(input_shape=(6, 6, 128), kernel_shape=(16, 3, 3, 128))
result = convolution(input_cube, kernels)
print(result[:,:,1].T)