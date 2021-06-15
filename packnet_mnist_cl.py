import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from nets import SmallerClassifier
from packnet import PackNet

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


print("Downloading Datasets")
trainloaders = []
testloaders = []

# MNIST
train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

# FashionMNIST
train = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

test_model = SmallerClassifier()
p_net = PackNet(model=test_model)

LR = .01
N_TRAIN_EPOCH = 3
N_FINE_TUNE_EPOCH = 1
loss = nn.NLLLoss()
sgd_optim = optim.SGD(test_model.parameters(), lr=LR)


#
# Training loop
#
print("Training Model")
for loader in trainloaders:
    # Train
    sgd_optim = optim.SGD(test_model.parameters(), lr=LR)  # Recreate optimizer on task switch
    for epoch in range(N_TRAIN_EPOCH):
        for img, cl in tqdm(loader):
            test_model.zero_grad()
            l = loss(test_model(img), cl)
            l.backward()
            p_net.training_mask()
            sgd_optim.step()
    # Prune
    p_net.prune(prune_quantile=.5)
    sgd_optim = optim.SGD(test_model.parameters(), lr=LR)

    # Fine-Tune
    for epoch in range(N_FINE_TUNE_EPOCH):
        for img, cl in tqdm(loader):
            test_model.zero_grad()
            l = loss(test_model(img), cl)
            l.backward()
            p_net.fine_tune_mask()
            sgd_optim.step()

    p_net.next_task()

# Test

print("\nTesting Model")
t1 = 0
accuracy = []

for img, cl in tqdm(testloaders[1]):
    pred = torch.argmax(test_model(img)[0])
    if pred != cl[0]:
        t1 += 1
accuracy.append(1.0 - (t1 / len(testloaders[1])))



# Results
print("")
for i, r in enumerate(accuracy):
    print(f'Accuracy on task {i + 1} : {r}')

# Accuracy trial 1: 0.5357000000000001