import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from nets import MnistClassifier, LightweightEncoder
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



test_model = MnistClassifier()
p_net = PackNet(model=test_model)

LR = .01
N_TRAIN_EPOCH = 1
N_FINE_TUNE_EPOCH = 1
loss = nn.NLLLoss()
sgd_optim = optim.SGD(test_model.parameters(), lr=LR)

print("Training")


for _ in range(N_TRAIN_EPOCH):
    for img, cl in tqdm(trainloaders[0]):
        test_model.zero_grad()
        l = loss(test_model(img), cl)
        l.backward()
        sgd_optim.step()


t1 = 0
for img, cl in tqdm(testloaders[0]):
    pred = torch.argmax(test_model(img)[0])
    if pred != cl[0]:
        t1 += 1
print(f'Accuracy before fine-tune: {1.0 - (t1 / len(testloaders[0]))}')

print("pruning")
p_net.prune(prune_quantile=.6)
print("done pruning")


sgd_optim = optim.SGD(test_model.parameters(), lr=LR)
for _ in range(N_FINE_TUNE_EPOCH):
    for img, cl in tqdm(trainloaders[0]):
        test_model.zero_grad()
        l = loss(test_model(img), cl)
        l.backward()

        p_net.fine_tune_mask()
        sgd_optim.step()


print("testing after fine-tune")

t1 = 0
for img, cl in tqdm(testloaders[0]):
    pred = torch.argmax(test_model(img)[0])
    if pred != cl[0]:
        t1 += 1
print(f'Accuracy after fine-tune: {1.0 - (t1 / len(testloaders[0]))}')