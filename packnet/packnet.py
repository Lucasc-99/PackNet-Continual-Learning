"""
Re-implementation of packnet continual learning method
"""

import torch
from torch import nn
from pytorch_lightning.callbacks import Callback


class PackNet(Callback):

    def __init__(self, n_tasks, prune_instructions, epoch_split):
        # Set up an array of quantiles for pruning procedure
        if isinstance(prune_instructions, list):  # if a list is passed in
            assert all(0 < i < 1 for i in prune_instructions)
            self.prune_instructions = prune_instructions

        else:  # if a float is passed in
            assert 0 < prune_instructions < 1
            self.prune_instructions = [prune_instructions] * (n_tasks - 1)

        self.PATH = None
        self.epoch_split = epoch_split
        self.current_task = 0
        self.n_tasks = n_tasks
        self.prune_instructions = prune_instructions
        self.masks = []  # 3-dimensions: task, layer, parameter mask
        self.mode = None

    def prune(self, model, prune_quantile):
        """
        Create task-specific mask and prune least relevant weights
        :param prune_quantile: The percentage of weights to prune as a decimal
        """
        # Calculate Quantile
        all_prunable = torch.tensor([])
        mask_idx = 0
        for mod in model.children():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        # get fixed weights for this layer
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)

                        for task in self.masks:
                            prev_mask |= task[mask_idx]

                        p = param_layer.masked_select(~prev_mask)
                        assert len(p) > 0, "No weights left to prune"
                        all_prunable = torch.cat((all_prunable.view(-1), p), -1)

                        mask_idx += 1

        cutoff = torch.quantile(torch.abs(all_prunable), q=prune_quantile)

        mask_idx = 0
        mask = []  # create mask for this task
        with torch.no_grad():
            for mod in model.children():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    for name, param_layer in mod.named_parameters():
                        if 'bias' not in name:
                            # get weight mask for this layer
                            prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)  # p
                            for task in self.masks:
                                prev_mask |= task[mask_idx]

                            curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                            curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                            # Zero non masked weights
                            param_layer *= (curr_mask | prev_mask)

                            mask.append(curr_mask)
                            mask_idx += 1

        self.masks.append(mask)

    def fine_tune_mask(self, model):
        """
        Zero the gradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        """
        assert len(self.masks) > self.current_task

        mask_idx = 0
        for mod in model.children():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        param_layer.grad *= self.masks[self.current_task][mask_idx]
                        mask_idx += 1

    def training_mask(self, model):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        """
        if len(self.masks) == 0:
            return

        mask_idx = 0
        for mod in model.children():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:
                        # get mask of weights from previous tasks
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        for task in self.masks:
                            prev_mask |= task[mask_idx]

                        # zero grad of previous fixed weights
                        param_layer.grad *= ~prev_mask

                        mask_idx += 1

    def fix_biases(self, model):
        """
        Fix the gradient of bias parameters
        """
        for mod in model.children():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' in name:
                        param_layer.requires_grad = False

    def fix_batch_norm(self, model):
        """
        Fix batch norm gain, bias, running mean and variance
        """
        for mod in model.children():
            if isinstance(mod, nn.BatchNorm2d):
                mod.affine = False
                for param_layer in mod.parameters():
                    param_layer.requires_grad = False

    def apply_eval_mask(self, model, task_idx):
        """
        Revert to network state for a specific task
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        """

        assert len(self.masks) > task_idx

        mask_idx = 0
        with torch.no_grad():
            for mod in model.children():
                if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                    for name, param_layer in mod.named_parameters():
                        if 'bias' not in name:

                            # get indices of all weights from previous masks
                            prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                            for i in range(0, task_idx + 1):
                                prev_mask |= self.masks[i][mask_idx]

                            # zero out all weights that are not in the mask for this task
                            param_layer *= prev_mask

                            mask_idx += 1

    def mask_remaining_params(self, model):
        """
        Create mask for remaining parameters
        """
        mask_idx = 0
        mask = []
        for mod in model.children():
            if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
                for name, param_layer in mod.named_parameters():
                    if 'bias' not in name:

                        # Get mask of weights from previous tasks
                        prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                        for task in self.masks:
                            prev_mask |= task[mask_idx]

                        # Create mask of remaining parameters
                        layer_mask = ~prev_mask
                        mask.append(layer_mask)

                        mask_idx += 1
        self.masks.append(mask)

    def total_epochs(self):
        return self.epoch_split[0] + self.epoch_split[1]

    def save_final_state(self, model, PATH='model_weights.pth'):
        """
        Save the final weights of the model after training
        :param model: pl_module
        :param PATH: The path to weights file
        """
        self.PATH = PATH
        torch.save(model.state_dict(), PATH)

    def load_final_state(self, model):
        """
        Load the final state of the model
        """
        model.load_state_dict(torch.load(self.PATH))

    def on_init_end(self, trainer):
        self.mode = 'train'

    def on_after_backward(self, trainer, pl_module):

        if self.mode == 'train':
            self.training_mask(pl_module)

        elif self.mode == 'fine_tune':
            self.fine_tune_mask(pl_module)

    def on_epoch_end(self, trainer, pl_module):

        if pl_module.current_epoch == self.epoch_split[0] - 1:  # Fine tune
            self.mode = 'fine_tune'
            if self.current_task == self.n_tasks - 1:
                self.mask_remaining_params(pl_module)
            else:
                self.prune(
                    model=pl_module,
                    prune_quantile=self.prune_instructions[self.current_task])

        elif pl_module.current_epoch == self.total_epochs() - 1:
            self.fix_biases(pl_module)  # Fix biases after first task
            self.fix_batch_norm(pl_module)  # Fix batch norm mean, var, and params
            self.current_task += 1
            self.save_final_state(pl_module)
            self.mode = 'train'