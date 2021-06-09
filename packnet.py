import torch
import torch.nn as nn


class PackNet(nn.Module):

    def __init__(self, model):
        super(PackNet, self).__init__()
        self.model = model
        self.current_task = 1
        self.masks = []

    def forward(self, x, task_id):
        return self.model(x)

    # Zero out percentage q of weights and create a mask
    def prune_weights(self, q):
        mask = []

        # Loop over every layer in model
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                layer_mask = set()
                cutoff = torch.quantile(input=torch.abs(torch.flatten(param_layer)), q=q)

                # Zero out least important weights
                param_layer.detach().apply_(lambda x: x * (abs(x) < cutoff))

                # Add most important weights to mask
                for i, val in enumerate(torch.flatten(param_layer)):
                    if val != 0:
                        layer_mask.add(i)

                mask.append(layer_mask)

        self.masks.append(mask)

    def switch_task(self, task_id):
        self.current_task = task_id
