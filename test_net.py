"""
Tests for net.py and packnet.py
"""
from net import MnistClassifier
from packnet import PackNet

m = MnistClassifier()
p_net = PackNet(model=m)


def test_net_init():
    assert(p_net is not None)
    assert(m is not None)


def test_prune_weights():
    p_net.prune_weights(q=.5)


test_net_init()
test_prune_weights()

