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
    p_net.fine_tune_mask()
    for p in test_model.parameters():
        for val in p.view(-1):
            if val.grad:
                print(val)


def test_get_fine_tune_params():
    p_net.get_fine_tune_params()
    print(list(test_model.parameters()))


#test_prune()
#test_fine_tune_mask()
 # can't get pytest to run on my conda env :/
test_get_fine_tune_params()