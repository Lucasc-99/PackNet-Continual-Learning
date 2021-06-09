import torch
import torch.nn as nn


class PackNet:

    def __init__(self, model, prune_quantile=.5):
        self.model = model
        self.q = prune_quantile
        self.masks = []

    # Zero out percentage q of weights and create a mask
    def prune_weights(self):
        mask = []

        # Loop over every layer in model
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                layer_mask = set()
                cutoff = torch.quantile(input=torch.abs(torch.flatten(param_layer)), q=self.q)

                # Zero out least important weights
                param_layer.detach().apply_(lambda x: x * (abs(x) >= cutoff))

                # Add most important weights to mask
                for i, val in enumerate(torch.flatten(param_layer)):
                    if val != 0:
                        layer_mask.add(i)

                mask.append(layer_mask)

        self.masks.append(mask)
