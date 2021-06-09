"""
Tests for net.py and packnet.py
"""
from net import MnistClassifier
from packnet import PackNet

m = MnistClassifier()
p_net = PackNet(model=m)
