"""
Train/Test split for evaluation of baseline performance on task incremental setting
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
from packnet.nets import MnistClassifier
import os
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

#
# Datasets
#
print("Downloading Datasets")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
trainloaders = []
testloaders = []

# MNIST
train = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

# FashionMNIST
train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

# KMNIST
train = datasets.KMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test = datasets.KMNIST(root=DATA_DIR, train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))


#
# Define model and hyperparameters
#
model = MnistClassifier()
LR = .01
N_EPOCH = 3
loss = nn.NLLLoss()
sgd_optim = optim.SGD(model.parameters(), lr=LR)


#
# Training loop
#
print("Training Model")
for loader in trainloaders:
    for _ in range(N_EPOCH):
        sgd_optim = optim.SGD(model.parameters(), lr=LR)  # Recreate optimizer on task switch
        for img, cl in tqdm(loader):
            model.zero_grad()
            l = loss(model(img), cl)
            l.backward()
            sgd_optim.step()

#
# Test Loop
#
accuracy = []
print("\nTesting Model")
for loader in testloaders:
    t1 = 0
    for img, cl in tqdm(loader):
        pred = torch.argmax(model(img)[0])
        if pred != cl[0]:
            t1 += 1
    accuracy.append(1.0 - (t1 / len(loader)))


# Results
print("")
for i, r in enumerate(accuracy):
    print(f'Accuracy on task {i + 1} : {r}')
