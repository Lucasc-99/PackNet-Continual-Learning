from net import MnistClassifier
from torch.utils.data import DataLoader

from continuum import ClassIncremental
from continuum.datasets import MNIST
from continuum.tasks import split_train_val
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
