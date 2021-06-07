import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistClassifier(nn.Module):

    def __init__(self):
        super(MnistClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.dense1 = nn.Linear(in_features=5760, out_features=2000)
        self.dense2 = nn.Linear(in_features=2000, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return F.log_softmax(x, dim=1)
