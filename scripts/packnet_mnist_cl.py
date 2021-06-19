"""
Train/Test split for evaluation of PackNet on task incremental setting
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from src.nets import SmallerClassifier, MnistClassifier
from src.packnet import PackNet

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


# KMNIST
# train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
# test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
# trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
# testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

test_model = MnistClassifier()
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
for i, loader in enumerate(trainloaders):
    # Train
    sgd_optim = optim.SGD(test_model.parameters(), lr=LR)  # Recreate optimizer on task switch
    for epoch in range(N_TRAIN_EPOCH):
        for img, cl in tqdm(loader):
            test_model.zero_grad()
            l = loss(test_model(img), cl)
            l.backward()
            p_net.training_mask()  # Zero grad previously fixed weights
            sgd_optim.step()

    if i == 0:
        p_net.prune(prune_quantile=.7)
    else:
        p_net.mask_remaining_params()

    sgd_optim = optim.SGD(test_model.parameters(), lr=LR)

    # Fine-Tune
    for epoch in range(N_FINE_TUNE_EPOCH):
        for img, cl in tqdm(loader):
            test_model.zero_grad()
            l = loss(test_model(img), cl)
            l.backward()
            p_net.fine_tune_mask()  # Zero grad for weights not being fine-tuned
            sgd_optim.step()

    p_net.fix_biases()  # Fix biases after first task
    p_net.next_task()

p_net.save_final_state()  # Save the final state of the model after training

# Test

print("\nTesting Model")

accuracy = []
for i, loader in enumerate(testloaders):
    t1 = 0
    p_net.apply_eval_mask(task_idx=i)

    for img, cl in tqdm(loader):
        pred = torch.argmax(test_model(img)[0])
        if pred != cl[0]:
            t1 += 1

    p_net.load_final_state()

    accuracy.append(1.0 - (t1 / len(loader)))

# Results

# NOTE: Using a classifier with a small amount of parameters
print("")
for i, r in enumerate(accuracy):
    print(f'Accuracy on task {i + 1} : {r}')

