# modules
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return torch.tensor([cell.accumulator for cell in self.cells])

class MACCell:
    def __init__(self, multipliers=64):
        self.weights = torch.empty(multipliers)
        self.accumulator = 0

    def __str__(self):
        return f"[{self.weights[0]:.2f}, {self.weights[1]:.2f}, ...]"

    def load(self, weights):
        self.weights = weights

    def mult(self, weights):
        self.accumulator = torch.sum(self.weights * weights)

class DirectConv2d():
    def __init__(self, stride=1, padding=0, dilation=1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_shape = (0, 0, 0, 0)

    def atomic(self, input_data, depth, y, x):
        # broadcast the input data and multiply and accumulate for each cell
        mac_array.broadcast(input_data[0, 64*depth:64*(depth+1), y, x])
        return mac_array.combine()

    def stripe(self, input_data, depth, ky, kx):
        result = torch.empty(self.output_shape)
        for oy in range(self.output_shape[2]):
            for ox in range(self.output_shape[3]):
                iy = oy + ky
                ix = ox + kx
                if iy < input_data.shape[2] and ix < input_data.shape[3]:
                    result[0, :, oy, ox] = self.atomic(input_data, depth, iy, ix)
        return result

    def block(self, input_data, kernels, depth):
        acc = torch.zeros(self.output_shape)
        for ky in range(kernels.shape[2]):
            for kx in range(kernels.shape[3]):
                load_kernels(kernels, depth, ky, kx)
                acc += self.stripe(input_data, depth, ky, kx)
        return acc

    def channel(self, input_data, kernels):
        block_ops = int((input_data.shape[1] + 63) / 64)
        blocks = torch.zeros(self.output_shape)
        for depth in range(block_ops):
            blocks += self.block(input_data, kernels, depth)
        return blocks

    def conv2d(self, input_cube, kernels):
        # (1, d, h, w)
        self.output_shape = (input_cube.shape[0],
                             kernels.shape[0],
                             int((self.padding * 2 + input_cube.shape[2] - (kernels.shape[2] - 1) * self.dilation - 1) / self.stride + 1),
                             int((self.padding * 2 + input_cube.shape[3] - (kernels.shape[3] - 1) * self.dilation - 1) / self.stride + 1))
        blocks = self.channel(input_cube, kernels)
        return blocks
    
def load_kernels(kernels, kd, ky, kx): # e.g. (x, y, z)
    # cache kernels into macs
    for i in range(mac_array.size):
        mac_array.cells[i].load(kernels[i, 64*kd:64*kd+64, ky, kx]) # load kernel weights into each mac cell

mac_array = MACArray(size=16, multipliers=64)
C = DirectConv2d(stride=1, padding=0, dilation=1)
input_cube = torch.rand((1, 128, 6, 6))
kernels = torch.rand((16, 128, 3, 3))

a = C.conv2d(input_cube, kernels)
b = F.conv2d(input_cube, kernels, padding=0)

print(torch.equal(a, b),
      F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item())
# print(a[0,0])
# print(b[0,0])
