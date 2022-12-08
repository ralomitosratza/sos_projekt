import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class DigitIdentifier:
    def __init__(self, train_data=None, test_data=None, batch_size=64, model_path=None, load=False, loss_fn=None,
                 optimizer=None, lr=0.1, momentum=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.train_data = self.get_data(train_data, Train=True)
        self.test_data = self.get_data(test_data, Train=False)
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

        self.model_path = model_path
        self.model = self.get_model(load=load)
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.optimizer = self.get_optimizer(optimizer, lr=lr, momentum=momentum)

    def get_model(self, load=False):
        if self.model_path is not None and os.path.isfile(self.model_path) and load is True:
            model = torch.load(self.model_path).to(self.device)
        else:
            model = neuralNet().to(self.device)
        return model

    def get_optimizer(self, optimizer, lr, momentum):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def train_model(self, epoch):
        self.model.train()
        train_loss, correct = 0, 0
        for batch_id, (data, target) in enumerate(self.train_dataloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            outpt = self.model(data)

            loss = self.loss_fn(outpt, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            correct += (outpt.argmax(1) == target).type(torch.float).sum().item()
            if batch_id % 200 == 0:
                print(f"current loss: {loss.item()} ")

        train_loss /= len(self.train_data)
        correct /= len(self.train_dataloader.dataset)
        print(f"Train Error (epoch: {epoch + 1}): Average accuracy: {(100 * correct)}%, Average loss: {train_loss} \n")

    def save_model(self):
        if self.model_path is not None:
            torch.save(self.model, self.model_path)

    @staticmethod
    def get_data(data, Train=True):
        if data is None:
            data = datasets.MNIST(
                root="data",
                train=Train,
                download=True,
                transform=ToTensor(),
            )
        return data

    @staticmethod
    def get_loss_fn(loss_fn):
        if loss_fn is None:
            loss_fn = F.nll_loss
        return loss_fn


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
        x = F.log_softmax(x, dim=1)  # softmax setzt die höchste Wahrscheinlichkeit auf 1, Rest auf 0
        return x



