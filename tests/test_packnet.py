"""
Tests for nets.py and packnet.py
"""
from torch import nn

from src.nets import SmallerClassifier
from src.packnet import PackNet
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




def test_prune():
    test_model = SmallerClassifier()
    p_net = PackNet(model=test_model)
    p_net.prune(prune_quantile=.7)
    p_net.fix_biases()
    p_net.next_task()

    total_masked = 0
    for task in p_net.masks:
        for layer_mask in task:
            total_masked += torch.count_nonzero(layer_mask.view(-1))
    print(f"Total Masked after prune 1: {total_masked}")

    p_net.prune(prune_quantile=.5)

    total_masked = 0
    for task in p_net.masks:
        for layer_mask in task:
            total_masked += torch.count_nonzero(layer_mask.view(-1))
    print(f"Total Masked after prune 2: {total_masked}")
    assert total_masked < 17335 # make sure we havent masked all the parameters
    assert len(p_net.masks) != 0


def test_fine_tune_mask():
    test_model = SmallerClassifier()
    p_net = PackNet(model=test_model)
    test_model.zero_grad()
    p_net.prune(prune_quantile=.9)
    for img, cl in trainloader:
        test_model.zero_grad()
        l = loss(test_model(img), cl)
        l.backward()
        break

    p_net.fine_tune_mask()

test_prune()

test_fine_tune_mask()

# can't get pytest to run on my conda env at the moment :(
