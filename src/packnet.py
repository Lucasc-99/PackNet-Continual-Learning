import torch


class PackNet:

    def __init__(self, model):
        self.PATH = None
        self.model = model
        self.current_task = 0
        self.masks = []  # 3-dimensions: task, layer, parameter mask

    def prune(self, prune_quantile):
        """
        Create task-specific mask and prune least relevant weights
        :return: Number of weights pruned
        """
        # Calculate Quantile
        all_prunable = torch.tensor([])
        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # get fixed weights for this layer
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                for task in self.masks:
                    prev_mask |= task[mask_idx]

                # Bug here?
                p = param_layer.masked_select(~prev_mask).view(-1)
                assert len(p) > 0, "No weights left to prune"
                all_prunable = torch.cat((all_prunable.view(-1), p), -1)

                mask_idx += 1

        cutoff = torch.quantile(torch.abs(all_prunable), q=prune_quantile)

        mask_idx = 0
        mask = []  # create mask for this task
        with torch.no_grad():
            for name, param_layer in self.model.named_parameters():

                if 'bias' not in name:

                    # get weight mask for this layer
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)  # p
                    for task in self.masks:
                        prev_mask |= task[mask_idx]

                    curr_mask = torch.abs(param_layer).ge(cutoff)  # q
                    curr_mask = torch.logical_and(curr_mask, ~prev_mask)  # (q & ~p)

                    # Zero non masked weights
                    param_layer *= curr_mask
                    # param_layer[curr_mask.eq(0)] = 0.0

                    mask.append(curr_mask)
                    mask_idx += 1

        self.masks.append(mask)

    def fine_tune_mask(self):
        """
        Zero the gradient of pruned weights this task as well as previously fixed weights
        Apply this mask before each optimizer step during fine-tuning
        :return: None
        """
        assert len(self.masks) > self.current_task

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # param_layer.grad[self.masks[self.current_task][mask_idx].eq(0)] = 0
                param_layer.grad *= self.masks[self.current_task][mask_idx]
                mask_idx += 1

    def training_mask(self):
        """
        Zero the gradient of only fixed weights for previous tasks
        Apply this mask after .backward() and before
        optimizer.step() at every batch of training a new task
        :return: None
        """
        if len(self.masks) == 0:
            return

        mask_idx = 0
        for name, param_layer in self.model.named_parameters():
            if 'bias' not in name:
                # get mask of weights from previous tasks
                prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                for task in self.masks:
                    prev_mask |= task[mask_idx]

                # zero grad of previous fixed weights
                param_layer.grad *= ~prev_mask  # param_layer.grad[prev_mask.ne(0)] = 0

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
        :param task_idx: the task id to be evaluated (0 - > n_tasks)
        :return: None
        """

        assert len(self.masks) > task_idx

        mask_idx = 0
        with torch.no_grad():
            for name, param_layer in self.model.named_parameters():
                if 'bias' not in name:

                    # get indices of all weights from previous masks
                    prev_mask = torch.zeros(param_layer.size(), dtype=torch.bool, requires_grad=False)
                    for i in range(0, task_idx + 1):
                        prev_mask |= self.masks[i][mask_idx]

                    # zero out all weights that are not in the mask for this task
                    param_layer *= prev_mask  # param_layer[prev_mask.eq(0)] = 0.0

                    mask_idx += 1

    def mask_remaining_params(self):
        """
        Create mask for remaining parameters
        :return: None
        """
        mask_idx = 0
        mask = []
        for name, param_layer in self.model.named_parameters():
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

    def save_final_state(self, PATH='model_weights.pth'):
        """
        Save the final weights of the model after training
        :param PATH: The path to weights file
        :return: None
        """
        self.PATH = PATH
        torch.save(self.model.state_dict(), PATH)

    def load_final_state(self):
        """
        Load the final state of the model
        :return:
        """
        self.model.load_state_dict(torch.load(self.PATH))

    def next_task(self):
        self.current_task += 1

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()
