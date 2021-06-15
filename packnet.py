import torch
from torch.autograd import Variable
import torch.nn as nn


class PackNet:

    def __init__(self, model):
        self.model = model
        self.mode = 'train'
        self.current_task = 0
        self.masks = []  # 3-dimensions: task, layer, parameter mask

    def prune(self, prune_quantile):
        """
        Create task-specific mask and prune least relevant weights
        :return: Number of weights pruned
        """

        mask = []  # create mask for this task
        weights_pruned = 0

        # Calculate Quantile

        all_prunable_params = torch.tensor([])
        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                flat = param_layer.view(-1)

                # get prunable weights for this layer
                prev_mask = set()
                for temp in self.masks:
                    if len(temp) > mask_idx:
                        prev_mask |= temp[mask_idx]

                prunable_weights = set([i for i in range(0, len(flat))]) - prev_mask

                # Concat layer parameters with all parameters tensor
                values = torch.index_select(torch.abs(flat), 0, torch.tensor(list(prunable_weights)))
                all_prunable_params = torch.cat((all_prunable_params, values), -1)
                mask_idx += 1

        cutoff = torch.quantile(input=all_prunable_params, q=prune_quantile)

        del all_prunable_params  # Garbage collection will do this for me?

        # Prune weights and create mask

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():

            if 'bias' not in name:
                flat = param_layer.view(-1)

                # get prunable weights for this layer
                prev_mask = set()
                for temp in self.masks:
                    if len(temp) > mask_idx:
                        prev_mask |= temp[mask_idx]

                # loop over layer parameters and prune
                curr_mask = set()
                with torch.no_grad():

                    for i, v in enumerate(flat):
                        if i not in prev_mask:
                            if torch.abs(v) >= cutoff:
                                curr_mask.add(i)
                            else:
                                v *= 0.0
                                weights_pruned += 1

                mask.append(curr_mask)
                mask_idx += 1

        self.masks.append(mask)
        return weights_pruned

    def next_task(self):
        self.current_task += 1

    def fine_tune_mask(self):
        """
        Zero the gradient of pruned weights as well as previously fixed weights
        Run this method before each optimizer step during fine-tuning
        :return: None
        """
        assert len(self.masks) > self.current_task

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:

                # get weights to be fine-tuned
                prev_mask = self.masks[self.current_task][mask_idx]

                # zero grad except for weights to fine-tune
                flat = param_layer.view(-1)
                with torch.no_grad():
                    for i, v in enumerate(flat):
                        if i not in prev_mask:
                            v.grad = Variable(torch.tensor(0.0))

                mask_idx += 1
