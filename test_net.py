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
    # w_1 = p_net.prune(q=.01)
    p_net.current_task = 1
    w_2 = p_net.prune(prune_quantile=1)
    #w_3 = p_net.prune(prune_quantile=.33)
    print(w_2)
    #print(w_3)

def test_mask_grad():

    p_net.masks = [[{0, 1, 2, 3, 4}]]

    layer_1 = list(test_model.parameters())
    conv1_params = torch.flatten(layer_1[0])
    conv1_params[0].grad = Variable(torch.tensor(1.0))
    p_net.mask_grad(0)
    print(conv1_params[0].grad)
    assert conv1_params[0].grad == 0


# test_mask_grad()
test_prune()  # can't get pytest to run on my conda env :/
