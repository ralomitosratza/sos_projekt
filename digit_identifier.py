import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

train_data = datasets.MNIST(
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

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class neuralNet(nn.Module):
    def __init__(self):
        super(neuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)                 # softmax setzt die höchste Wahrscheinlichkeit auf 1, Rest auf 0
        return x


model = neuralNet().to(device)
if os.path.isfile('neuralNetworks/digit_identifier_nn.pt'):
    model = torch.load('neuralNetworks/digit_identifier_nn.pt')
    model = model.to(device)

loss_fn = F.nll_loss
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)


def train(epoch):
    model.train()
    train_loss, correct = 0, 0
    for batch_id, (data, target) in enumerate(train_dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outpt = model(data)

        loss = loss_fn(outpt, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (outpt.argmax(1) == target).type(torch.float).sum().item()
        if batch_id % 200 == 0:
            print(f"current loss: {loss.item()} ")

    train_loss /= len(train_data)
    correct /= len(train_dataloader.dataset)
    print(f"Train Error (epoch: {epoch + 1}): Average accuracy: {(100 * correct)}%, Average loss: {train_loss} \n")


for epoch in range(1, 10):
    train(epoch)

torch.save(model, 'neuralNetworks/digit_identifier_nn.pt')
