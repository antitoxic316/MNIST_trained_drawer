from torch import nn

class CNNmodel(nn.Module):
    def __init__(self, in_shape, out_shape) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_shape,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4
            ),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=4,
                stride=4
            ),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64,
                      out_features=out_shape),
            nn.Softmax()
        )
    
    def forward(self, x):
        return self.classifier(self.conv_block_1(x))
