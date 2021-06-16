import torch
from torch.autograd import Variable
import torch.nn as nn
import tqdm


class PackNet:

    def __init__(self, model):
        self.PATH = None
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
        with torch.no_grad():
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

                        # Bottleneck here on dense layers
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

    def fine_tune_mask(self):
        """
        Zero the gradient of pruned weights as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        :return: None
        """
        if len(self.masks) <= self.current_task:
            return

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # get weights to be fine-tuned
                prev_mask = self.masks[self.current_task][mask_idx]

                # zero grad except for weights to fine-tune
                for i, v in enumerate(param_layer.grad.view(-1)):
                    if i not in prev_mask and v:
                        v.zero_()
                mask_idx += 1

    def training_mask(self):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        :return: None
        """
        assert len(self.masks) == self.current_task

        if len(self.masks) == 0:
            return

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # get indices of weights from previous masks
                prev_mask = set()

                for m in self.masks:
                    assert len(m) > mask_idx
                    prev_mask |= m[mask_idx]

                # zero grad of previous fixed weights
                for i, v in enumerate(param_layer.grad.view(-1)):
                    if i in prev_mask and v:
                        v.zero_()
                mask_idx += 1

    def fix_biases(self):
        """
        Fix the gradient of bias parameters
        :return: None
        """
        for name, param_layer in self.model.named_parameters():
            if 'bias' in name:
                param_layer.requires_grad = False

    def apply_eval_mask(self, task_idx):
        """
        Revert to network state for a specific task
        :param task_idx:
        :return:
        """

        assert len(self.masks) > task_idx

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # get indices of weights from previous masks
                prev_mask = set()
                for i in range(0, task_idx + 1):
                    prev_mask |= self.masks[i][mask_idx]

                # zero out all weights that are not in the mask for this task
                with torch.no_grad():
                    for i, v in enumerate(param_layer.view(-1)):
                        if i not in prev_mask:
                            v *= 0.0
                mask_idx += 1

    def save_final_state(self, PATH='model_weights.pth'):
        self.PATH = PATH
        torch.save(self.model.state_dict(), PATH)

    def load_final_state(self):
        self.model.load_state_dict(torch.load(self.PATH))

    def next_task(self):
        self.current_task += 1

    # Unimplemented methods
    def get_fine_tune_params(self):
        """
        Get parameters for fine-tuning (should be much faster than fine_tune_mask)
        :return: An iterable with only parameters for fine-tuning
        """
        # Ideally should modify the self.model.parameters() iterable in-place,
        # keeping only the parameters that will be fine-tuned and passed to the optimizer

        # This will save compute for running
        # fine_tune_mask on every batch.
        # is this even possible?
