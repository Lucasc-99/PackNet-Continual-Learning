"""
Wrapper for PackNet integration into the Sequoia Research Tree Library
"""

from packnet.packnet import PackNet
from sequoia import Method
from sequoia.settings.sl import TaskIncrementalSLSetting
from pytorch_lightning import Trainer
import torch

# Eventually:
# from sequoia.methods import BaseMethod


class PackNetMethod(Method, target_setting=TaskIncrementalSLSetting):
    def __init__(self, model, prune_instructions, epoch_split, n_tasks=None):
        self.model = model
        self.p_net = PackNet(n_tasks=n_tasks,  # This gets set in configure
                             prune_instructions=prune_instructions,
                             epoch_split=epoch_split)

        self.p_net.current_task = -1  # Because Sequoia calls task switch before first fit

    def configure(self, s):
        self.p_net.n_tasks = s.nb_tasks
        self.p_net.config_instructions()

    def fit(self, train_env, valid_env):
        trainer = Trainer(callbacks=[self.p_net], max_epochs=self.p_net.total_epochs())
        trainer.fit(model=self.model, train_dataloader=train_env)

    def get_actions(self,
                    observations,
                    observation_space):
        with torch.no_grad():
            y_pred = torch.argmax(self.model(observations.x), dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id):
        self.p_net.current_task = task_id
