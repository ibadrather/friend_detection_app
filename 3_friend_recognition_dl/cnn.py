import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(
        self, input_channels, num_targets, kernel_size=3, stride=1, dropout=0.15
    ):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 32, kernel_size=kernel_size, stride=stride, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.output = nn.Sequential(
            nn.Linear(256, 512),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.Dropout(dropout),
            nn.Linear(512, num_targets),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.mean(3).mean(2)
        # x = x.view(x.size(0), -1)

        x = self.output(x)
        return x


def test():
    from torchinfo import summary
    # Testing
    data = torch.randn(1, 3, 224, 224)
    model = CNN(3, 2)
    print(model(data))

    summary(model, input_size=(1, 3, 224, 224))


if __name__ == "__main__":
    test()
