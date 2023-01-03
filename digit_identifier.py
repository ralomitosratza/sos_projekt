import pickle
import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from pynvml.smi import nvidia_smi


class DigitIdentifier:
    def __init__(self, train_data=None, test_data=None, epochs=None, batch_size=64, load=False, csv_index=0,
                 forward_dict=None, loss_fn=None, optimizer=None, lr=0.1, momentum=0.8, weight_decay=0.0001, info=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.info = info
        if self.info:
            print(f"Using {self.device} device")
        self.epochs = epochs
        self.train_data = self.get_data(train_data, train=True)
        self.test_data = self.get_data(test_data, train=False)
        self.batch_size = batch_size
        self.forward_dict = forward_dict
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
        self.model = self.get_model(forward_dict=forward_dict, load=load, csv_index=csv_index)
        self.best_model = None
        if load is False:
            self.loss_fn = self.get_loss_fn(loss_fn)
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.optimizer = self.get_optimizer(optimizer)
        self.needed_time = None
        self.memory_total = None
        self.memory_used = None
        self.memory_free = None
        self.average_accuracy_train = None
        self.average_accuracy_test = 0
        self.average_loss_train = None
        self.average_loss_test = None

    def get_model(self, forward_dict=None, load=False, csv_index=0):
        path = f"models/digit_identifier{csv_index}.pt"
        if load is True and os.path.isfile(path) is True:
            model = torch.load(path).to(self.device)
        else:
            model = NeuralNet(forward_dict=forward_dict).to(self.device)
        return model

    def get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer == 'optim.SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer == 'optim.Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer == 'optim.Adadelta':
            optimizer = optim.Adadelta(self.model.parameters())
        elif optimizer == 'optim.Adagrad':
            optimizer = optim.Adagrad(self.model.parameters())
        elif optimizer == 'optim.AdamW':
            optimizer = optim.AdamW(self.model.parameters())
        elif optimizer == 'optim.SparseAdam':
            optimizer = optim.SparseAdam(self.model.parameters())
        elif optimizer == 'optim.Adamax':
            optimizer = optim.Adamax(self.model.parameters())
        elif optimizer == 'optim.ASGD':
            optimizer = optim.ASGD(self.model.parameters())
        elif optimizer == 'optim.LBFGS':
            optimizer = optim.LBFGS(self.model.parameters())
        elif optimizer == 'optim.NAdam':
            optimizer = optim.NAdam(self.model.parameters())
        elif optimizer == 'optim.RAdam':
            optimizer = optim.RAdam(self.model.parameters())
        elif optimizer == 'optim.RMSprop':
            optimizer = optim.RMSprop(self.model.parameters())
        elif optimizer == 'optim.Rprop':
            optimizer = optim.Rprop(self.model.parameters())
        return optimizer

    def train_model(self, stop_counter_max=3):
        start_time = time.time()
        correct = None
        train_loss = None
        average_accuracy_test_best = -1
        epoch = -1
        stop_counter = 0
        while stop_counter <= stop_counter_max:
            epoch += 1
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
                if batch_id % 200 == 0 and self.info:
                    print(f"current loss: {loss.item()} ")
            train_loss /= len(self.train_data)
            correct /= len(self.train_dataloader.dataset)
            if self.info:
                print(f"Train Error (epoch: {epoch + 1}): Average accuracy: {(100 * correct)}%, "
                      f"Average loss: {train_loss}\n")
            self.test_model()
            if average_accuracy_test_best < self.average_accuracy_test:
                average_accuracy_test_best = self.average_accuracy_test
                self.save_all(model=True, rest=False)
                self.epochs = epoch + 1
                stop_counter = 0
            else:
                stop_counter += 1

        self.average_accuracy_test = average_accuracy_test_best
        self.needed_time = time.time() - start_time
        self.memory_total = self.get_memory_usage('total')
        self.memory_used = self.get_memory_usage('used')
        self.memory_free = self.memory_total - self.memory_used
        self.average_accuracy_train = 100 * correct
        self.average_loss_train = train_loss
        self.save_all(model=False, rest=True)

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
        if self.info:
            print(f"Test Error: Accuracy: {(100 * correct)}%, Average loss: {test_loss} \n")
        self.average_accuracy_test = 100 * correct
        self.average_loss_test = test_loss

    def save_all(self, model=True, rest=True):
        path = 'panda_tables/runs.csv'
        if os.path.exists(path):
            num = len(pd.read_csv(path))
        else:
            num = 0
        if rest is True:
            df = pd.DataFrame({'average_accuracy_test': self.average_accuracy_test,
                               'average_loss_test': self.average_loss_test,
                               'needed_time': self.needed_time, 'memory_used': self.memory_used/self.memory_total,
                               'memory_total': self.memory_total, 'epochs': self.epochs, 'batch_size': self.batch_size,
                               'loss_function': self.loss_fn, 'optimizer': self.optimizer, 'learning_rate': self.lr,
                               'momentum': self.momentum}, index=[1])
            df.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
            forward_dict = open(f'dictionarys/forward_dictionary{num}.pkl', 'wb')
            pickle.dump(self.forward_dict, forward_dict)
        if model is True:
            torch.save(self.model, f"models/digit_identifier{num}.pt")

    def try_model(self, show=5):
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
    def __init__(self, forward_dict=None):
        super(NeuralNet, self).__init__()
        picture_size = 28
        last_output = 0
        self.forward_dict = forward_dict
        self.layer_list = nn.ModuleList()
        for step in forward_dict:
            if forward_dict[step]['action'] == 'layer':
                layer, picture_size, last_output = self.get_layer(forward_dict[step], picture_size, last_output)
                self.layer_list.append(layer)
            elif forward_dict[step]['action'] == 'f.max_pool2d':
                picture_size = int(picture_size/forward_dict[step]['kernel_size'])
        self.view_dim2 = picture_size*picture_size*last_output

    def forward(self, x):
        i = 0
        for step in self.forward_dict:
            if self.forward_dict[step]['action'] == 'layer':
                x = self.layer_list[i](x)
                i += 1

            elif self.forward_dict[step]['action'] == 'f.max_pool2d':
                x = f.max_pool2d(x, kernel_size=self.forward_dict[step]['kernel_size'])
            elif self.forward_dict[step]['action'] == 'f.relu':
                x = f.relu(x)
            elif self.forward_dict[step]['action'] == 'view':
                x = x.view(self.forward_dict[step]['dim1'], self.view_dim2)
            elif self.forward_dict[step]['action'] == 'f.log_softmax':
                x = f.log_softmax(x, dim=self.forward_dict[step]['dim'])
            else:
                print('Failure while forward.')
        return x

    @staticmethod
    def get_layer(layer, size, out):
        if layer['layer'] == 'conv2d':
            size = size - (layer['kernel_size'] - 1)
            out = layer['out']
            return nn.Conv2d(layer['in'], layer['out'], kernel_size=layer['kernel_size']), size, out
        elif layer['layer'] == 'conv_dropout2d':
            return nn.Dropout2d(), size, out
        elif layer['layer'] == 'linear':
            if layer['in'] == 0:
                return nn.Linear(size*size*out, layer['out']), size, out
            else:
                return nn.Linear(layer['in'], layer['out']), size, out
        else:
            return None
