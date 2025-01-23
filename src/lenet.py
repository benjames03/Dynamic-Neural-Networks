import random
from torch import nn, no_grad, int32, float32
import conv

class LeNet(nn.Module):
    """
    A LeNet architecture model with two convolutions and three linear layers
    """

    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.linear1 = nn.Linear(208, 120)
        self.linear2 = nn.Linear(120, 86)
        self.linear3 = nn.Linear(86, 10)

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        
        return out
    
class SimLeNet(nn.Module):
    """
    A simulated LeNet modelling MAC hardware units
    """

    def __init__(self):
        super(SimLeNet, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = conv.SimConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = conv.SimConv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.linear1 = conv.SimLinear(208, 120)
        self.linear2 = conv.SimLinear(120, 86)
        self.linear3 = conv.SimLinear(86, 10)

    def load_state_dict(self, state_dict, strict = True, assign = False):
        new_state_dict = {}
        for key, value in state_dict.items():
            if "linear" in key:
                if "weight" in key:
                    new_key = key.replace("weight", "conv.weight")
                    new_value = value.view(value.shape[0], 1, 1, value.shape[1])
                else:
                    new_key = key.replace("bias", "conv.bias")
                    new_value = value
                new_state_dict[new_key] = new_value
            else:
                new_state_dict[key] = value
        return super().load_state_dict(new_state_dict, strict, assign)
    
    def inject_faults(self, faults):
        with no_grad():
            for i in range(faults):
                bit = random.randint(0, 31)
                mult = random.randint(0, self.conv1.weight.size(1)-1)
                ker = random.randint(0, self.conv1.weight.size(0)-1)
                self.conv1.weight[ker, mult, :, :] = (self.conv1.weight[ker, mult, :, :].view(int32) | (1 << bit)).view(float32)
                
                mult = random.randint(0, self.conv2.weight.size(1)-1)
                ker = random.randint(0, self.conv2.weight.size(0)-1)
                self.conv2.weight[ker, mult, :, :] = (self.conv2.weight[ker, mult, :, :].view(int32) | (1 << bit)).view(float32)
                
                mult = random.randint(0, self.linear1.conv.weight.size(1)-1)
                ker = random.randint(0, self.linear1.conv.weight.size(0)-1)
                self.linear1.conv.weight[ker, mult, :, :] = (self.linear1.conv.weight[ker, mult, :, :].view(int32) | (1 << bit)).view(float32)
                
                mult = random.randint(0, self.linear2.conv.weight.size(1)-1)
                ker = random.randint(0, self.linear2.conv.weight.size(0)-1)
                self.linear2.conv.weight[ker, mult, :, :] = (self.linear2.conv.weight[ker, mult, :, :].view(int32) | (1 << bit)).view(float32)
                
                mult = random.randint(0, self.linear3.conv.weight.size(1)-1)
                ker = random.randint(0, self.linear3.conv.weight.size(0)-1)
                self.linear3.conv.weight[ker, mult, :, :] = (self.linear3.conv.weight[ker, mult, :, :].view(int32) | (1 << bit)).view(float32)

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        
        return out