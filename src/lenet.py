from torch import nn, no_grad
import conv

class LeNet(nn.Module):
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
    def __init__(self):
        super(SimLeNet, self).__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0)
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