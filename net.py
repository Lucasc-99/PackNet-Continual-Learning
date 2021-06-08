from dataclasses import dataclass
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sequoia.settings import Method, Setting
from torch.utils.data import DataLoader
from sequoia.settings.passive.cl import TaskIncrementalSetting
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    PassiveEnvironment,
    Results,
    Rewards,
)


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


class SimpleMNISTClassifierMethod(Method, target_setting=TaskIncrementalSetting):
    def __init__(self):
        self.learning_rate = .001
        self.model = MnistClassifier()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.loss = nn.NLLLoss()

    def fit(self, train_env: DataLoader, valid_env):
        for observation, reward in train_env:
            img = observation.x
            y_pred = self.model(img)
            if reward is None:
                # Send action to env to get reward
                reward = train_env.send(y_pred)
            cl = reward.y
            self.model.zero_grad()
            l = self.loss(y_pred, cl)
            l.backward()
            self.optimizer.step()

    def get_actions(self, observations, observation_space):
        return self.model(observations)  # Do I need to reshape observations?

