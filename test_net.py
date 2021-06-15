"""
Tests for nets.py and packnet.py
"""
from torch import nn
from torch.autograd import Variable

from nets import LightweightEncoder, SmallerClassifier
from packnet import PackNet
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


# MNIST
train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

loss = nn.NLLLoss()

test_model = SmallerClassifier()
p_net = PackNet(model=test_model)


def test_prune():
    w_1 = p_net.prune(prune_quantile=.33)
    assert w_1 != 0
    print(w_1)


def test_fine_tune_mask():
    test_model.zero_grad()
    print(p_net.prune(prune_quantile=.2))
    for img, cl in trainloader:
        test_model.zero_grad()
        l = loss(test_model(img), cl)
        l.backward()
        break

    p_net.fine_tune_mask()

# test_prune()
test_fine_tune_mask()
# can't get pytest to run on my conda env :/
