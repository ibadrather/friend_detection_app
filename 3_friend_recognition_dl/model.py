import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()

        self.expansion = 4  # out 4x in as seen in the table

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()

        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class Resnet2D(nn.Module):  # [3, 4, 6, 3]
    def __init__(self, block: Block, layers, data_channels: int, output_size: int):
        """
        data_channels: it can be number of image channels or imu data chanenls
        output_size: it can be number of classes
        """
        super(Resnet2D, self).__init__()
        self.in_channels = 64

        self.expansion = 4

        self.conv1 = nn.Conv2d(
            data_channels, self.in_channels, kernel_size=7, stride=2, padding=3
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        # self.layer1 = ...
        # self.layer2 = ...
        # We will write a function _make_layer to do this

        self.layer1 = self._make_layer(
            block=block, num_residual_blocks=layers[0], out_channels=64, stride=1
        )

        self.layer2 = self._make_layer(
            block=block, num_residual_blocks=layers[1], out_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block=block, num_residual_blocks=layers[2], out_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block=block, num_residual_blocks=layers[3], out_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * self.expansion, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc(x)
        return x

    # def _make_layer(self, block: Block, num_residual_blocks: list,
    #                 out_channels: int, stride: int):

    def _make_layer(
        self, block: Block, num_residual_blocks: int, out_channels: int, stride: int
    ):

        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * self.expansion:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

        layers.append(
            block(
                in_channels=self.in_channels,
                out_channels=out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            )
        )
        # Updating in channels for next
        self.in_channels = out_channels * self.expansion

        # -1 because we already computed one residual block
        for i in range(num_residual_blocks - 1):
            layers.append(
                block(self.in_channels, out_channels)
            )  # 256 -> 64, 54*4(256) again

        return nn.Sequential(*layers)  # unpack list


def Resnet12(data_channels, output_size=10):
    return Resnet2D(
        block=Block,
        layers=[1, 1, 1, 1],
        data_channels=data_channels,
        output_size=output_size,
    )


def Resnet18(data_channels, output_size=10):
    return Resnet2D(
        block=Block,
        layers=[1, 1, 2, 2],
        data_channels=data_channels,
        output_size=output_size,
    )


def Resnet50(data_channels, output_size=10):
    return Resnet2D(
        block=Block,
        layers=[3, 4, 6, 3],
        data_channels=data_channels,
        output_size=output_size,
    )


def Resnet101(data_channels, output_size=10):
    return Resnet2D(
        block=Block,
        layers=[3, 4, 23, 3],
        data_channels=data_channels,
        output_size=output_size,
    )


def Resnet152(data_channels, output_size=10):
    return Resnet2D(
        block=Block,
        layers=[3, 8, 36, 3],
        data_channels=data_channels,
        output_size=output_size,
    )


class SimpleCNN(nn.Module):
    def __init__(self, data_channels, output_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(data_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)

        # x = x.reshape(x.shape[0], 128, -1)
        x = x.mean(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).squeeze(0)
        return x


class NetExample(nn.Module):
    def __init__(self, in_channels=3, out_classes=2):
        super(NetExample, self).__init__()

        ##################################
        # ENCODER PART OF THE NETWORK
        # B x 3 x 224 x 224
        # Number of kernels: K
        # Size of kernel:    k x k
        # Stride:            1 x 1
        # Padding:           SAME
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=8, kernel_size=(5, 5), padding=2
        )
        # How many parameters are in conv1 layer?
        # 5 x 5 x 3 x 4 = 300
        # B x 4 x 224 x 224
        self.pool1 = nn.MaxPool2d(2, 2)
        # B x 4 x 112 x 112
        self.conv2 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=(5, 5), padding=2
        )
        # 5 x 5 x 4 x 8 = 3200
        # B x 8 x 112 x 112
        self.pool2 = nn.MaxPool2d(2, 2)
        # B x 8 x 56 x 56
        self.conv3 = nn.Conv2d(
            in_channels=8, out_channels=8, kernel_size=(5, 5), padding=2
        )
        # 5 x 5 x 8 x 8 = 6400
        # B x 8 x 56 x 56
        self.pool3 = nn.MaxPool2d(2, 2)
        # B x 8 x 28 x 28
        # TOTAL N. OF PARAMETERS: 9900
        ###################################

        ###################################
        # CLASSIFIER PART OF THE NETWORK
        # 8 x 28 x 28 = 6272 Input
        # Flatten [1,...,6272]
        # Parameters ? 313600
        # 50 Output
        # B x 6272
        self.hidden = nn.Linear(in_features=8 * 28 * 28, out_features=50)
        # B x 50
        self.final = nn.Linear(in_features=50, out_features=out_classes)
        # B x 2
        ###################################

    def forward(self, x):
        ##################################
        # ENCODER PART
        # First Convolutional Block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # Second Convolutional Block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Third Convolutional Block
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        ##################################

        ##################################
        # CLASSIFIER PART
        x = x.view(-1, 8 * 28 * 28)
        x = self.hidden(x)
        x = F.relu(x)
        x = self.final(x)
        ##################################

        return x


class Net(nn.Module):
    def __init__(self, in_channels=3, out_features=2):
        super(Net, self).__init__()

        # input has shape B x in_channels x 224 x 224
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=32, kernel_size=(3, 3), padding=1
        )
        # output has shape B x 32 x 224 x 224

        self.pool1 = nn.MaxPool2d(2, 2)
        # output has shape B x 32 x 112 x 112

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1
        )
        # output has shape B x 64 x 112 x 112

        self.pool2 = nn.MaxPool2d(2, 2)
        # output has shape B x 64 x 56 x 56

        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1
        )
        # output has shape B x 128 x 56 x 56

        self.pool3 = nn.MaxPool2d(2, 2)
        # output has shape B x 128 x 28 x 28

        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1
        )
        # output has shape B x 256 x 28 x 28

        self.pool4 = nn.MaxPool2d(2, 2)
        # output has shape B x 256 x 14 x 14

        self.fc1 = nn.Linear(256 * 14 * 14, 128)
        # output has shape B x 128

        self.fc2 = nn.Linear(128, out_features=out_features)
        # output has shape B x out_features

        # builtin in CrossEntropyLoss
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.pool4(x)

        x = x.view(-1, 256 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # builtin in CrossEntropyLoss
        # x = self.softmax(x)
        return x


def test():
    from torchinfo import summary
    data = torch.randn(1, 3, 224, 224)
    model = NetExample()

    summary(model, input_data=data)


if __name__ == "__main__":
    test()
    