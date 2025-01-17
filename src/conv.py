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
        for cell in self.cells:
            cell.mult(input_weights)

    def combine(self):
        return torch.tensor([cell.accumulator for cell in self.cells])

class MACCell:
    def __init__(self, multipliers=64):
        self.weights = torch.zeros(multipliers)
        self.accumulator = 0

    def __str__(self):
        return f"[{self.weights[0]:.2f}, {self.weights[1]:.2f}, ...]"

    def load(self, weights):
        self.weights.zero_()
        self.weights[:len(weights)] = weights

    def mult(self, weights):
        self.accumulator = torch.sum(self.weights[:len(weights)] * weights)

class DirectConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.mac_array = MACArray(size=out_channels, multipliers=64)
        self.output_shape = (0, 0, 0, 0)

    def load_kernels(self, kd, ky, kx): # e.g. (z, y, x)
        # cache kernels into macs
        for i in range(self.mac_array.size):
            self.mac_array.cells[i].load(self.weight[i, 64*kd:64*(kd+1), ky, kx]) # load kernel weights into each mac cell

    def atomic(self, input_data, depth, y, x):
        # broadcast the input data and multiply and accumulate for each cell
        self.mac_array.broadcast(input_data[0, 64*depth:64*(depth+1), y, x])
        return self.mac_array.combine()

    def stripe(self, input_data, depth, ky, kx):
        result = torch.zeros(self.output_shape)
        for oy in range(self.output_shape[2]):
            for ox in range(self.output_shape[3]):
                iy = oy * self.stride[0] + ky - self.padding[0]
                ix = ox * self.stride[1] + kx - self.padding[1]
                if 0 <= iy and iy < input_data.shape[2] and \
                   0 <= ix and ix < input_data.shape[3]:
                    result[0, :, oy, ox] = self.atomic(input_data, depth, iy, ix)
        return result

    def block(self, input_data, depth):
        acc = torch.zeros(self.output_shape)
        for ky in range(self.weight.shape[2]):
            for kx in range(self.weight.shape[3]):
                self.load_kernels(depth, ky, kx)
                acc += self.stripe(input_data, depth, ky, kx)
        return acc

    def channel(self, input_data):
        block_ops = int((input_data.shape[1] + 63) / 64)
        blocks = torch.zeros(self.output_shape)
        for depth in range(block_ops):
            blocks += self.block(input_data, depth)
        return blocks

    def conv2d(self, input):
        # (1, d, h, w)
        self.output_shape = (input.shape[0],
                             self.weight.shape[0],
                             int((self.padding[0] * 2 + input.shape[2] - (self.weight.shape[2] - 1) * self.dilation[0] - 1) / self.stride[0] + 1),
                             int((self.padding[1] * 2 + input.shape[3] - (self.weight.shape[3] - 1) * self.dilation[1] - 1) / self.stride[1] + 1))
        out = self.channel(input)
        out += self.bias.view(1, out.shape[1], 1, 1)
        return out
    
    def forward(self, input):
        return self.conv2d(input)
