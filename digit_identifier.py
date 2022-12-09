import os
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from pynvml.smi import nvidia_smi
import time


class DigitIdentifier:
    def __init__(self, train_data=None, test_data=None, batch_size=64, model_path=None, load=False, loss_fn=None,
                 optimizer=None, lr=0.1, momentum=0.8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.train_data = self.get_data(train_data, train=True)
        self.test_data = self.get_data(test_data, train=False)
        self.batch_size = batch_size

        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)

        self.model_path = model_path
        self.model = self.get_model(load=load)
        self.loss_fn = self.get_loss_fn(loss_fn)
        self.optimizer = self.get_optimizer(optimizer, lr=lr, momentum=momentum)
        self.needed_time = None
        self.memory_total = None
        self.memory_used = None
        self.memory_free = None

    def get_model(self, load=False):
        if self.model_path is not None and os.path.isfile(self.model_path) and load is True:
            model = torch.load(self.model_path).to(self.device)
        else:
            model = NeuralNet().to(self.device)
        return model

    def get_optimizer(self, optimizer, lr, momentum):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return optimizer

    def train_model(self, epochs):
        start_time = time.time()
        for epoch in range(epochs):
            self.model.train()
            train_loss, correct = 0, 0
            for batch_id, (data, target) in enumerate(self.train_dataloader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(data)

                loss = self.loss_fn(prediction, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
                if batch_id % 200 == 0:
                    print(f"current loss: {loss.item()} ")
            train_loss /= len(self.train_data)
            correct /= len(self.train_dataloader.dataset)
            print(correct)
            print(f"Train Error (epoch: {epoch + 1}): Average accuracy: {(100 * correct)}%, Average loss: {train_loss}\n")

        self.memory_total = self.get_memory_usage('total')
        self.memory_used = self.get_memory_usage('used')
        self.memory_free = self.memory_total - self.memory_used
        self.needed_time = time.time() - start_time
        self.save_model()

    def test_model(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                test_loss += self.loss_fn(prediction, target).item()
                correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
        test_loss /= len(self.test_dataloader)
        correct /= len(self.test_dataloader.dataset)
        print(f"Test Error: Accuracy: {(100 * correct)}%, Average loss: {test_loss} \n")

    def save_model(self):
        if self.model_path is not None:
            torch.save(self.model, self.model_path)

    def show_results(self, show=5):
        shown = 0
        self.model.eval()
        images, labels = next(iter(self.test_dataloader))
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                for i in range(data.size()[0]):
                    if prediction[i].argmax(0) == target[i]:
                        plt.title(f'Prediction: {prediction[i].argmax(0)} -> Correct!')
                        plt.imshow(images[i].reshape(28, 28), cmap="summer")
                        plt.show()
                    else:
                        plt.title(f'Prediction: {prediction[i].argmax(0)} -> Not correct!')
                        plt.imshow(images[i].reshape(28, 28), cmap="autumn")
                        plt.show()
                    shown += 1
                    if shown >= show:
                        break
                if shown >= show:
                    break

    @staticmethod
    def get_data(data, train=True):
        if data is None:
            data = datasets.MNIST(
                root="data",
                train=train,
                download=True,
                transform=ToTensor(),
            )
        return data

    @staticmethod
    def get_loss_fn(loss_fn):
        if loss_fn is None:
            loss_fn = f.nll_loss
        return loss_fn

    @staticmethod
    def get_memory_usage(option):
        return nvidia_smi.getInstance().DeviceQuery('memory.' + option)['gpu'][0]['fb_memory_usage'][option]


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv_dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 60)
        self.fc2 = nn.Linear(60, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = self.conv2(x)
        x = self.conv_dropout(x)
        x = f.max_pool2d(x, 2)
        x = f.relu(x)
        x = x.view(-1, 320)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        x = f.log_softmax(x, dim=1)  # softmax setzt die höchste Wahrscheinlichkeit auf 1, Rest auf 0
        return x
