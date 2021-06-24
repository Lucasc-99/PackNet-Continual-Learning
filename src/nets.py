"""
Networks used in /scripts and /tests
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistClassifier(nn.Module):

    def __init__(self, input_channels=1):
        super(MnistClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=5)
        self.dense1 = nn.Linear(in_features=5760, out_features=2000)
        self.dense2 = nn.Linear(in_features=2000, out_features=10)

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x, dim=1)


class SmallerClassifier(nn.Module):

    def __init__(self, input_channels=1):
        super(SmallerClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=5)
        self.norm_layer = nn.BatchNorm2d(num_features=1728, affine=True)
        self.dense1 = nn.Linear(in_features=1728, out_features=10)

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.norm_layer(self.conv1(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        return F.log_softmax(x, dim=1)


class SequentialClassifier(nn.Module):
    """
    The purpose of this is to test different module structures
    """
    def __init__(self):

        super(SequentialClassifier, self).__init__()

        # Model Taken from: https://androidkt.com/convolutional-neural-network-using-sequential-model-in-pytorch/
        self.model_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten())

        self.model_classifier = nn.Sequential(
            nn.Linear(2303, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))

    def forward(self, x):
        return self.model_classifier(self.model_encoder(x))
