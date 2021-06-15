import torch
from torch.autograd import Variable
import torch.nn as nn


class PackNet:

    def __init__(self, model):
        self.model = model
        self.mode = 'train'
        self.current_task = 0
        self.masks = []  # 3-dimensions: task, layer, parameter mask

    def prune(self, q):
        """
        Create task-specific mask and prune model
        :return: Number of weights pruned
        """

        mask = []  # create mask for this task
        weights_pruned = 0
        mask_idx = 0

        for m in self.model.modules():
            for j, (name, param_layer) in enumerate(m.named_parameters()):

                if 'bias' not in name:
                    layer_mask = set()
                    cutoff = torch.quantile(input=torch.abs(torch.flatten(param_layer)), q=q)

                    with torch.no_grad():
                        flat = param_layer.view(-1)

                        for i, v in enumerate(flat):
                            prunable = True

                            for temp in self.masks:
                                if len(temp) > mask_idx and i in temp[mask_idx]:
                                    prunable = False

                            if prunable:
                                if v >= cutoff:
                                    layer_mask.add(i)
                                else:
                                    v *= 0.0
                                    weights_pruned += 1

                    mask.append(layer_mask)
                    mask_idx += 1

        self.masks.append(mask)
        return weights_pruned
