"""
Wrapper for PackNet integration into the Sequoia Research Tree Library
"""

from src.packnet import PackNet
from src.nets import MnistClassifier
from sequoia import Method, TaskIncrementalSetting
from torch import optim
from tqdm import tqdm
import torch.nn as nn


class PackNetMethod(Method, target_setting=TaskIncrementalSetting):

    def __init__(self, model, N_TRAIN_EPOCH=5, N_FINE_TUNE_EPOCH=2):
        self.mode = 'train'
        self.model = model
        self.p_net = PackNet(self.model)
        self.N_TRAIN = N_TRAIN_EPOCH
        self.N_TUNE = N_FINE_TUNE_EPOCH

    def fit(self, train_env, valid_env):
        print("Fit called")
        # can i assume all of these samples are of the same task?
        # can i just iterate these parameters like train/test loaders?

        # how do i separate training and fine tuning?

        LR = 0.01
        loss = nn.NLLLoss()

        # Train
        sgd_optim = optim.SGD(self.model.parameters(), lr=LR)  # Recreate optimizer on task switch
        for epoch in range(self.N_TRAIN):
            for observation, reward in tqdm(train_env):
                self.model.zero_grad()
                l = loss(self.model(observation.x), reward.y)
                l.backward()
                self.p_net.training_mask()  # Zero grad previously fixed weights
                sgd_optim.step()

        self.p_net.prune(prune_quantile=.7)

        sgd_optim = optim.SGD(self.model.parameters(), lr=LR)

        # Fine-Tune
        for epoch in range(self.N_TUNE):
            for observation, reward in tqdm(train_env):
                self.model.zero_grad()
                l = loss(self.model(observation.x), reward.y)
                l.backward()
                self.p_net.fine_tune_mask()  # Zero grad for weights not being fine-tuned
                sgd_optim.step()

        self.p_net.fix_biases()  # Fix biases after first task
        self.p_net.fix_batch_norm()  # Fix batch norm mean, var, and params

        self.p_net.save_final_state()  # Save the final state of the model after training

    def get_actions(self,
                    observations,
                    observation_space):
        """
        Assume that every observation in observations has the same task
        """
        print("GET ACTIONS")
        assert observations.task_labels[0] == observations.task_labels[1]
        assert self.p_net.current_task == observations.task_labels[0]
        return self.model(observations.x)

    def on_task_switch(self, task_id):

        if self.mode == 'train' and len(self.p_net.masks) > 0:
            self.p_net.next_task()

        elif self.mode == 'test':
            self.p_net.load_final_state()
            self.p_net.apply_eval_mask(task_idx=task_id)
            self.p_net.current_task = task_id


setting = TaskIncrementalSetting(
    dataset="mnist",
    increment=2
)

m = MnistClassifier(input_channels=3)
my_method = PackNetMethod(model=m)
results = setting.apply(my_method)
