from torch import nn
from torchvision import models


def get_big_model(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dp(x)

        return x


class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.block1 = CNNBlock(3, 16, (3, 3))
        self.pool1 = nn.MaxPool2d((2, 2))

        self.block2 = CNNBlock(16, 32, (3, 3))
        self.pool2 = nn.MaxPool2d((2, 2))

        self.block3 = CNNBlock(32, 64, (3, 3))
        self.pool3 = nn.MaxPool2d((2, 2))

        self.block4 = CNNBlock(64, 128, (3, 3))
        self.pool4 = nn.MaxPool2d((2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(128*6*6, 128)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = self.pool3(x)

        x = self.block4(x)
        x = self.pool4(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dp(x)
        x = self.fc2(x)

        return x
