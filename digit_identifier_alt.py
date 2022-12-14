import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import pynvml.smi as nvsmi
import logging
from datetime import datetime


logging.basicConfig(filename="mircos_log.log", level=logging.INFO)
logging.warning(f"{datetime.now().isoformat()}: Error on url:")
nvsmi.nvmlInit()
handle = nvsmi.nvmlDeviceGetHandleByIndex(0)

training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device: {torch.cuda.get_device_name()}")
# res = nvsmi.nvmlDeviceGetUtilizationRates(handle)
# print(nvsmi.nvmlDeviceGetMemoryInfo(handle))
# print(nvsmi.nvmlDeviceGetName(handle))
# print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
nvsmiss = nvsmi.nvidia_smi.getInstance()
print(nvsmiss.DeviceQuery('memory.free, memory.total'))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
# model = torch.load('neuralNetworks/digit_identifier_alt_nn.pt')
# model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(dataloader, model, loss_fn, optimizer):
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 200 == 0:
            print(f"current loss: {loss.item()} ")
    train_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Train Error: Average accuracy: {(100 * correct)}%, Average loss: {train_loss} \n")


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)
    print(f"Test Error: Accuracy: {(100 * correct)}%, Average loss: {test_loss} \n")


epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

torch.save(model, 'neuralNetworks/digit_identifier_alt_nn.pt')
print("Done!")
