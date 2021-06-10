"""
Tests for net.py and packnet.py
"""
from net import MnistClassifier
from packnet import PackNet

m = MnistClassifier()
p_net = PackNet(model=m)


def test_prune_weights():
    p_net.prune_weights()

def test_mask_grad():
    pass


test_prune_weights()