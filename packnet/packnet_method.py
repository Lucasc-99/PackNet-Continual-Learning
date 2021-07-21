"""
Wrapper for PackNet integration into the Sequoia Research Tree Library
"""

from packnet.packnet import PackNet
from packnet.nets import MnistClassifier
from sequoia import Method
from sequoia.settings.sl import TaskIncrementalSLSetting
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch
# Eventually:
# from sequoia.methods import BaseMethod


class PackNetMethod(Method, target_setting=TaskIncrementalSLSetting):
    def __init__(self, model: nn.Module, N_TRAIN_EPOCH=5, N_FINE_TUNE_EPOCH=2, prune_quantile=.7, LR: float = 0.01):
        self.mode = 'train'
        self.model = model
        self.p_net = PackNet(self.model)
        self.p_quantile = prune_quantile
        self.N_TRAIN = N_TRAIN_EPOCH
        self.N_TUNE = N_FINE_TUNE_EPOCH
        self.LR = LR
        self.p_net.current_task = -1  # Because Sequoia calls task switch before first fit
        self.loss = nn.CrossEntropyLoss()

    def configure(self, setting: TaskIncrementalSLSetting):
        # TODO: Use this to configure the method before it gets trained/evaluated on the given Setting.
        self.nb_tasks = setting.nb_tasks

    def fit(self, train_env, valid_env):
        # can i assume all of these samples are of the same task?
        # can i just iterate these parameters like train/test loaders?

        # how do i separate training and fine tuning?

        # Train
        # Recreate optimizer on task switch

        # if torch lightning module, just use .training_step()

        sgd_optim = optim.SGD(self.model.parameters(), lr=self.LR)
        for epoch in range(self.N_TRAIN):
            for observation, reward in tqdm(train_env):
                # if torch lightning module, just use .training_step()
                self.model.zero_grad()
                logits = self.model(observation.x)
                l = self.loss(logits, reward.y)

                l.backward()
                self.p_net.training_mask()  # Zero grad previously fixed weights
                sgd_optim.step()

        self.p_net.prune(prune_quantile=self.p_quantile)

        sgd_optim = optim.SGD(self.model.parameters(), lr=LR)

        # Fine-Tune
        for epoch in range(self.N_TUNE):
            for observation, reward in tqdm(train_env):
                self.model.zero_grad()
                l = self.loss(self.model(observation.x), reward.y)
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

        '''assert observations.task_labels[0] == observations.task_labels[-1]
        assert self.p_net.current_task == observations.task_labels[0]'''

        with torch.no_grad():
            y_pred = torch.argmax(self.model(observations.x), dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id):
        if len(self.p_net.masks) > task_id:
            self.p_net.load_final_state()
            self.p_net.apply_eval_mask(task_idx=task_id)
        self.p_net.current_task = task_id


if __name__ == "__main__":
    setting = TaskIncrementalSLSetting(
        dataset="mnist",
        increment=2
    )

    m = MnistClassifier(input_channels=3)
    my_method = PackNetMethod(model=m)
    results = setting.apply(my_method)
    # results.make_plots()
