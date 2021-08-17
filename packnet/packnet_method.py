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
    def __init__(self, model, prune_instructions, epoch_split):
        self.model = model
        self.prune_instructions = prune_instructions
        self.epoch_split = epoch_split
        self.p_net: PackNet  # This gets set in configure

    def configure(self, setting: TaskIncrementalSLSetting):
        self.p_net = PackNet(
            n_tasks=setting.nb_tasks,  
            prune_instructions=self.prune_instructions,
            epoch_split=self.epoch_split,
        )
        # Because Sequoia calls task switch before first fit
        self.p_net.current_task = -1
        self.p_net.config_instructions()

    def fit(self, train_env, valid_env):
        trainer = Trainer(callbacks=[self.p_net], max_epochs=self.p_net.total_epochs())
        trainer.fit(model=self.model, train_dataloader=train_env)
        self.p_net.save_final_state(self.model)

    def get_actions(self,
                    observations,
                    action_space):
        with torch.no_grad():
            logits = self.model(observations.x.to(self.model.device))
            y_pred = logits.argmax(dim=-1)
        return self.target_setting.Actions(y_pred)

    def on_task_switch(self, task_id):

        if len(self.p_net.masks) > task_id:
            self.p_net.load_final_state(model=self.model)
            self.p_net.apply_eval_mask(task_idx=task_id, model=self.model)
        self.p_net.current_task = task_id
