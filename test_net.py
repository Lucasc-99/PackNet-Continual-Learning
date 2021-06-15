"""
Tests for nets.py and packnet.py
"""
from torch import nn

from nets import MnistClassifier, LightweightEncoder
from packnet import PackNet
import torch
from torch.autograd import Variable

test_model = LightweightEncoder()
p_net = PackNet(model=test_model)


def test_prune():
    w_1 = p_net.prune(prune_quantile=.33)
    test_model.backward()
    assert w_1 != 0
    print(w_1)


def test_fine_tune_mask():
    p_net.prune(prune_quantile=.1)
    p_net.fine_tune_mask()
    with torch.no_grad():
        mask_idx = 0
        for name, param_layer in p_net.model.named_parameters():
            if 'bias' not in name:
                flat = param_layer.view(-1)
                for i, v in enumerate(flat):
                    v.requires_grad = False
                    print(v.requires_grad)
# test_prune()
test_fine_tune_mask()
# can't get pytest to run on my conda env :/
