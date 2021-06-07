"""
A simple train/test split on MNIST digits using the model in net.py

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from net import MnistClassifier

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])


model = MnistClassifier()


train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)

test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(test, batch_size=1, shuffle=True)

LR = .01
N_EPOCH = 5

loss = nn.NLLLoss()
sgd_optim = optim.SGD(model.parameters(), lr=LR)

for epoch in range(1, N_EPOCH + 1):
    running_loss = 0
    for img, cl in trainloader:
        model.zero_grad()
        l = loss(model(img), cl)
        running_loss += l
        l.backward()
        sgd_optim.step()
    print(f'Running loss after epoch {epoch} is {running_loss}')

type_1 = 0
for img, cl in testloader:
    pred = torch.argmax(model(img)[0])
    if pred != cl[0]:
        type_1 += 1
print(f'type-1 error rate is: {type_1/len(testloader)}')
