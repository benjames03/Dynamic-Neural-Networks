from torch import nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=30, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=30, out_channels=13, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3),
            nn.Flatten(),
            nn.Linear(208, 120),
            nn.ReLU(),
            nn.Linear(120, 86),
            nn.ReLU(),
            nn.Linear(86, 10)
        )

    def forward(self, x):
        out = self.linear_relu_stack(x)
        return out