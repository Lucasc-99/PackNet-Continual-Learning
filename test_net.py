"""
Tests for net.py
"""
from net import MnistClassifier, PackNet
import pytest


def test_net_init():
    m = MnistClassifier()
    p_net = PackNet(model=m)

    assert(p_net is not None)
    assert(m is not None)


def test_prune_weights():
    pass


def test_task_switch():
    pass


test_net_init()

