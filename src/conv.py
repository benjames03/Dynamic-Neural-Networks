import torch
import torch.nn as nn
import torch.nn.functional as F

class MACArray:
    """
    Simulates a set of MAC hardware units

    size (int): The number of MAC units
    multipliers (int): The number of multipliers in each cell
    """

    def __init__(self, size=16, multipliers=64):
        self.size = size
        self.cells = [MACCell(multipliers) for _ in range(size)]
        self.multipliers = multipliers
    
    def __str__(self):
        """Prints the cells and their values"""
        str = ""
        for i, cell in enumerate(self.cells):
            str += f"Cell {i:02} - {cell}\n"
        return str

    def broadcast(self, weights):
        """Broadcasts weights to all MAC cells and multiplies them"""
        for cell in self.cells:
            cell.mult(weights)

    def combine(self):
        """Returns array of all current cell accumulators"""
        return torch.tensor([cell.accumulator for cell in self.cells])

class MACCell:
    """
    Simulates a single MAC cell

    multipliers (int): The number of multipliers in the cell
    """

    def __init__(self, multipliers=64):
        self.weights = torch.zeros(multipliers)
        self.accumulator = 0

    def __str__(self):
        """Prints first two values of the cell"""
        return f"[{self.weights[0]:.2f}, {self.weights[1]:.2f}, ...]"

    def load(self, weights):
        """Loads kernel weights into the cell"""
        self.weights.zero_()
        self.weights[:len(weights)] = weights

    def mult(self, weights):
        """Multiplies broadcasted input with held kernel weights"""
        self.accumulator = torch.sum(self.weights[:len(weights)] * weights)

class SimConv2d(nn.Conv2d):
    """
    A sub class of Conv2d that performs the forward pass using simulated hardware

    in_channels (int): Number of channels in the input image (cube depth)
    out_channels (int): Number of channels produced by the convolution (kernels)
    kernel_size (int or tuple): Size of the convolving kernel
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(SimConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.mac_array = MACArray(size=out_channels, multipliers=64)
        self.output_shape = (0, 0, 0, 0)

    def load_kernels(self, kd, ky, kx): # e.g. (z, y, x)
        """Cache each kernel slice into a MAC"""
        for i in range(self.mac_array.size):
            self.mac_array.cells[i].load(self.weight[i, 64*kd:64*(kd+1), ky, kx])

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

    def conv2d(self, input): # assuming batches are of size 1
        # (1, d, h, w)
        self.output_shape = (1,
                             self.weight.shape[0],
                             int((self.padding[0] * 2 + input.shape[2] - (self.weight.shape[2] - 1) * self.dilation[0] - 1) / self.stride[0] + 1),
                             int((self.padding[1] * 2 + input.shape[3] - (self.weight.shape[3] - 1) * self.dilation[1] - 1) / self.stride[1] + 1))
        # if (batch_size := input.shape[0]) != 1:
        #     out = torch.zeros((batch_size, *self.output_shape[1:]))
        #     for b in range(batch_size):
        #         out[b] = self.channel(input[b].unsqueeze(0))
        #     out += self.bias.view(1, -1, 1, 1).expand(out.shape[0], -1, out.shape[2], out.shape[3])
        # else:
        out = self.channel(input)
        out += self.bias.view(1, -1, 1, 1)
        return out
    
    def forward(self, input):
        return self.conv2d(input)
    
class SimLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = nn.Conv2d(in_channels=1, out_channels=out_features, kernel_size=(1, in_features), stride=1, padding=0,)

    def forward(self, input):
        input = input.view(1, 1, -1)
        out = self.conv(input)
        out = out.view(out.shape[1], out.shape[0])
        return out
