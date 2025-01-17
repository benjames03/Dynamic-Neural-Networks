from torch import nn
import conv

class LeNet(nn.Module):
    def __init__(self, simulating=False):
        super(LeNet, self).__init__()
        self.relu = nn.ReLU()
        # self.conv1 = conv.DirectConv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0) \
        #              if simulating else \
        #              nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = conv.DirectConv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0) \
                     if simulating else \
                     nn.Conv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.linear1 = nn.Linear(208, 120)
        self.linear2 = nn.Linear(120, 86)
        self.linear3 = nn.Linear(86, 10)

    def forward(self, x):
        out = self.conv1(x)
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