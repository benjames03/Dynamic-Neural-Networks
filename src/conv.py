import torch
import torch.nn as nn
import torch.nn.functional as F

class MACArray:
    """
    Simulates a set of MAC hardware units

    size (int): The number of MAC units
    multipliers (int): The number of multipliers in each cell
    """

    def __init__(self, batches=25, size=16, multipliers=64):
        self.multipliers = multipliers
        self.cells = torch.empty((size, multipliers))
        self.accumulators = torch.empty(batches, size)
        self.fault_macs = []
        self.fault_mults = []
        self.fault_masks = []

    def load(self, weights): # (size, multipliers)
        """Load the kernel weights into the array with zero padding if necessary"""
        weights = F.pad(weights, (0, self.multipliers-weights.size(1)), value=0)
        self.cells.copy_(weights)

    def broadcast(self, weights): # (batches, multipliers)
        """Broadcasts weights to all MAC cells and multiplies them"""
        padded_weights = F.pad(weights, (0, self.multipliers-weights.size(1)), value=0)
        inter_weights = padded_weights[:, :, None] * self.cells.T[None, :, :]
        if (self.fault_macs != []):
            inter_weights[:, self.fault_mults, self.fault_macs] = (inter_weights[:, self.fault_mults, self.fault_macs].view(torch.int32) | self.fault_masks).view(torch.float32)
        self.accumulators.copy_(inter_weights.sum(dim=1))

class SimConv2d(nn.Conv2d):
    """
    A sub class of Conv2d that performs the forward convolution pass using simulated hardware

    in_channels (int): Number of channels in the input image (cube depth)
    out_channels (int): Number of channels produced by the convolution (kernels)
    kernel_size (int or tuple): Size of the convolving kernel
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: 0
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(SimConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.mac_array = MACArray(batches=50, size=out_channels, multipliers=64)
        self.output_shape = (0, 0, 0, 0)
    
    def inject_faults(self, faults):
        fault_macs, fault_mults, fault_masks = [], [], []
        for (macu, mult, bit) in faults:
            if macu < self.weight.shape[0] and mult < self.weight.shape[1]:
                for mac in range(macu, self.weight.shape[0], 16):
                    fault_macs.append(mac)
                    fault_mults.append(mult)
                    if bit == 31:
                        fault_masks.append(-2**31)
                    else:
                        fault_masks.append(1 << bit)
        fault_macs = torch.tensor(fault_macs, dtype=torch.int32)
        fault_mults = torch.tensor(fault_mults, dtype=torch.int32)
        fault_masks = torch.tensor(fault_masks, dtype=torch.int32)
        # with torch.no_grad():
        #     self.weight[fault_macs, fault_mults, :, :] = (self.weight[fault_macs, fault_mults, :, :].view(torch.int32) | fault_masks.view(-1, 1, 1).expand(-1, self.weight.shape[2], self.weight.shape[3])).view(torch.float32)

        self.mac_array.fault_macs = fault_macs
        self.mac_array.fault_mults = fault_mults
        self.mac_array.fault_masks = fault_masks

    def load_kernels(self, kd, ky, kx): # e.g. (z, y, x)
        """Cache each kernel slice into a MAC cell"""
        self.mac_array.load(self.weight[:, self.mac_array.multipliers*kd:self.mac_array.multipliers*(kd+1), ky, kx])

    def atomic(self, input_data, depth, y, x):
        """Broadcast the input slice and multiply and accumulate for each cell"""
        self.mac_array.broadcast(input_data[:, self.mac_array.multipliers*depth:self.mac_array.multipliers*(depth+1), y, x])
        return self.mac_array.accumulators

    def stripe(self, input_data, depth, ky, kx):
        """Slide the sub-kernel position over the cube face"""
        result = torch.zeros(self.output_shape)
        for oy in range(self.output_shape[2]):
            for ox in range(self.output_shape[3]):
                iy = oy * self.stride[0] + ky - self.padding[0]
                ix = ox * self.stride[1] + kx - self.padding[1]
                if 0 <= iy and iy < input_data.shape[2] and \
                   0 <= ix and ix < input_data.shape[3]:
                    result[:, :, oy, ox] = self.atomic(input_data, depth, iy, ix)
        return result

    def block(self, input_data, depth):
        """Compute a stripe for all sub-kernels"""
        acc = torch.zeros(self.output_shape)
        for ky in range(self.weight.shape[2]):
            for kx in range(self.weight.shape[3]):
                self.load_kernels(depth, ky, kx)
                acc += self.stripe(input_data, depth, ky, kx)
        return acc

    def channel(self, input_data):
        """Compute a block for each depth in the cube"""
        block_ops = int((input_data.shape[1] + self.mac_array.multipliers-1) / self.mac_array.multipliers)
        blocks = torch.zeros(self.output_shape)
        for depth in range(block_ops):
            blocks += self.block(input_data, depth)
        return blocks

    def conv2d(self, input): # input must be (1, d, h, w)
        """Compute a forward convolution pass"""
        if len(input.shape) == 3: 
            input = input.unsqueeze(0)
        self.output_shape = (input.shape[0],
                             self.weight.shape[0],
                             int((self.padding[0] * 2 + input.shape[2] - (self.weight.shape[2] - 1) * self.dilation[0] - 1) / self.stride[0] + 1),
                             int((self.padding[1] * 2 + input.shape[3] - (self.weight.shape[3] - 1) * self.dilation[1] - 1) / self.stride[1] + 1))
        out = self.channel(input)
        out += self.bias.view(1, -1, 1, 1)
        return out
    
    def forward(self, input):
        return self.conv2d(input)
    
class SimLinear(nn.Module):
    """
    A linear wrapping layer that uses an underlying convolution forward pass

    in_features (int): Number of input neurons
    out_features (int): Number of output neurons
    """

    def __init__(self, in_features, out_features):
        super(SimLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv = SimConv2d(in_channels=1, out_channels=out_features, kernel_size=(1, in_features), stride=1, padding=0)

    def forward(self, input):
        """Compute forward pass for the simulated linear layer"""
        input = input.view(input.shape[0], 1, 1, -1)
        out = self.conv(input)
        out = out.view(out.shape[0], -1)
        return out
