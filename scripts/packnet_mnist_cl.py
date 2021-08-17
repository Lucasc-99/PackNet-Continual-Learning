"""
Train/Test split for evaluation of PackNet on task incremental setting
"""
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

from packnet.nets import MnistClassifier, SequentialClassifier
from packnet.packnet import PackNet

from pytorch_lightning import Trainer

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

train = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
trainloaders.append(torch.utils.data.DataLoader(train, batch_size=64, shuffle=True))
testloaders.append(torch.utils.data.DataLoader(test, batch_size=1, shuffle=True))

# Init model
test_model = MnistClassifier()

# Init
train_epochs = 3
tune_epochs = 1
p_net = PackNet(n_tasks=3,
                prune_instructions=[.7, .7],
                epoch_split=(train_epochs, tune_epochs))


#
# Training loop
#
print("Training Model")
for i, loader in enumerate(trainloaders):
    trainer = Trainer(callbacks=[p_net], max_epochs=p_net.total_epochs())
    trainer.fit(model=test_model, train_dataloader=loader)
    p_net.save_final_state(test_model)
    p_net.current_task += 1

# Test

print("\nTesting Model")

accuracy = []
for i, loader in enumerate(testloaders):
    t1 = 0
    p_net.apply_eval_mask(task_idx=i, model=test_model)

    for img, cl in tqdm(loader):
        pred = torch.argmax(test_model(img)[0])
        if pred != cl[0]:
            t1 += 1

    p_net.load_final_state(model=test_model)

    accuracy.append(1.0 - (t1 / len(loader)))

# Results

# NOTE: Using a classifier with a small amount of parameters
print("")
for i, r in enumerate(accuracy):
    print(f'Accuracy on task {i + 1} : {r}')