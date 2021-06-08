from net import MnistClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sequoia.settings import Method, Setting
from torch.utils.data import DataLoader
from sequoia.settings.passive.cl import TaskIncrementalSetting
from dataclasses import dataclass
import gym
from sequoia.settings.passive.cl.objects import (
    Actions,
    Environment,
    Observations,
    PassiveEnvironment,
    Results,
    Rewards,
)

import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


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
