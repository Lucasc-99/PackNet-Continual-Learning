"""
Networks used in /scripts and /tests
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Tuple, Optional
from sequoia.settings.sl.continual import Observations, Rewards, Environment


class MnistClassifier(pl.LightningModule):
    """
    Example classifier, used in packnet_mnist_cl.py and packnet_sequoia.py
    """

    def __init__(self, input_channels=1):
        super(MnistClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=5)
        self.dense1 = nn.Linear(in_features=5760, out_features=2000)
        self.dense2 = nn.Linear(in_features=2000, out_features=10)
        self.trainer: pl.Trainer

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        return F.log_softmax(x, dim=-1)  # Apply log softmax and return

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        return F.nll_loss(out, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

class SequentialMnistClassifier(pl.LightningModule):
    """
    Example classifier, used in packnet_mnist_cl.py and packnet_sequoia.py
    """

    def __init__(self, input_channels=1):
        super(SequentialMnistClassifier, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=5760, out_features=2000),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=10),
            nn.ReLU()
        )
        self.trainer: pl.Trainer

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))

        return F.log_softmax(x, dim=-1)  # Apply log softmax and return

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        return F.nll_loss(out, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class SequoiaClassifier(pl.LightningModule):
    """
    Example classifier, used in packnet_mnist_cl.py and packnet_sequoia.py
    """

    def __init__(self, input_channels=1):
        super(SequoiaClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=5)
        self.dense1 = nn.Linear(in_features=5760, out_features=2000)
        self.dense2 = nn.Linear(in_features=2000, out_features=10)
        self.trainer: pl.Trainer

    def forward(self, x):
        """
        :param x: 1x28x28 tensor representing MNIST image
        :return: logits, 10 classes
        """
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))  # Final logits

        return F.log_softmax(x, dim=-1)  # Apply log softmax and return

    def training_step(self, batch: Tuple[Observations, Optional[Rewards]], batch_idx):
        observations, rewards = batch
        x = observations.x

        logits = self(x)
        y_pred = logits.argmax(-1)
        if rewards is None:
            # NOTE: See the pl_example.py in `sequoia/examples/basic/pl_example.py` for
            # more info about when this might happen.
            environment: Environment = self.trainer.request_dataloader("train")
            rewards = environment.send(y_pred)

        assert rewards is not None
        y = rewards.y
        accuracy = (y_pred == y).int().sum().div(len(y))
        self.log("train/accuracy", accuracy, prog_bar=True)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class SmallerSequoiaClassifier(pl.LightningModule):

    def __init__(self, input_channels=1):
        super(SmallerSequoiaClassifier, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=3, kernel_size=5)
        self.norm_layer = nn.BatchNorm2d(num_features=3, affine=True)
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

    def training_step(self, batch, batchidx):
        observations, rewards = batch
        x = observations.x

        logits = self(x)
        y_pred = logits.argmax(-1)
        if rewards is None:
            # NOTE: See the pl_example.py in `sequoia/examples/basic/pl_example.py` for
            # more info about when this might happen.
            environment: Environment = self.trainer.request_dataloader("train")
            rewards = environment.send(y_pred)

        assert rewards is not None
        y = rewards.y
        accuracy = (y_pred == y).int().sum().div(len(y))
        self.log("train/accuracy", accuracy, prog_bar=True)
        loss = F.nll_loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class SequentialClassifier(pl.LightningModule):
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

    def training_step(self, batch, batchidx):
        x, y = batch
        return F.nll_loss(self(x), y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)
