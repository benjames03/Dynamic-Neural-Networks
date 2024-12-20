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
        self.output_shape = (0, 0, 0)

    def atomic(self, input_data, depth, pos):
        # broadcast the input data and multiply and accumulate for each cell
        mac_array.broadcast(input_data[pos][64*depth:64*(depth+1)])
        return mac_array.combine()

    def stripe(self, input_data, depth, kpos):
        result = torch.empty(self.output_shape)
        for oy in range(self.output_shape[1]):
            for ox in range(self.output_shape[0]):
                ix = ox + kpos[0]
                iy = oy + kpos[1]
                if ix < input_data.shape[0] and iy < input_data.shape[1]:
                    result[ox, oy] = self.atomic(input_data, depth, (ix, iy))
        return result

    def block(self, input_data, kernels, depth):
        acc = torch.zeros(self.output_shape)
        for y in range(kernels.shape[2]):
            for x in range(kernels.shape[1]):
                load_kernels(kernels, (x, y, depth))
                acc += self.stripe(input_data, depth, (x, y))
        return acc

    def channel(self, input_data, kernels):
        block_ops = int((input_data.shape[-1] + 63) / 64)
        blocks = torch.zeros(self.output_shape)
        for depth in range(block_ops):
            blocks += self.block(input_data, kernels, depth)
        return blocks

    def conv2d(self, input_cube, kernels):
        self.output_shape = (int((self.padding * 2 + input_cube.shape[0] - (kernels.shape[1] - 1) * self.dilation - 1) / self.stride + 1),
                             int((self.padding *2 + input_cube.shape[1] - (kernels.shape[2] - 1) * self.dilation - 1) / self.stride + 1),
                             kernels.shape[0])
        blocks = self.channel(input_cube, kernels)
        return blocks
    
def load_kernels(kernels, pos): # e.g. (x, y, z)
    # cache kernels into macs
    p = 64 * pos[-1]
    for i in range(mac_array.size):
        mac_array.cells[i].load(kernels[i][pos[:-1]][p:p+64]) # load kernel weights into each mac cell

mac_array = MACArray(size=16, multipliers=64)
C = DirectConv2d(stride=1, padding=0, dilation=1)
input_cube = torch.rand((6, 6, 128))
kernels = torch.rand((16, 3, 3, 128))
result = C.conv2d(input_cube, kernels)
# print(torch.equal(C.conv2d(input_cube, kernels), F.conv2d(input_cube, kernels)))
# print(F.conv2d(input_cube, kernels).shape)
