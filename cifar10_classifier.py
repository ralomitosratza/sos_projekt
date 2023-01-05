import os
import time
import torch
import matplotlib.pyplot as plt
from model_functions import NeuralNet, get_loss_fn, get_memory_usage, save_all, get_optimizer
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class Cifar10Classifier:
    def __init__(self, train_data=None, test_data=None, batch_size=64, load=False, csv_index=0, forward_dict=None,
                 loss_fn=None, optimizer=None, lr=0.1, momentum=0.8, weight_decay=0.0001, info=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.device = 'cpu'
        self.info = info
        self.csv_path = 'panda_tables/runs_cifar10_classifier.csv'
        self.dict_path = 'dictionarys/cifar10_classifier/forward_dictionary'
        self.model_path = 'models/cifar10_classifier/cifar10_classifier'
        if self.info:
            print(f"Using {self.device} device")
        self.epochs = None
        self.train_data = self.get_data(train_data, train=True, download=False)
        self.test_data = self.get_data(test_data, train=False, download=False)
        self.batch_size = batch_size
        self.forward_dict = forward_dict
        self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size)
        self.model = self.get_model(forward_dict=forward_dict, load=load, csv_index=csv_index)
        self.best_model = None
        if load is False:
            self.loss_fn = get_loss_fn(loss_fn)
            self.lr = lr
            self.momentum = momentum
            self.weight_decay = weight_decay
            self.optimizer = get_optimizer(model=self, optimizer=optimizer)
        self.needed_time = None
        self.memory_total = None
        self.memory_used = None
        self.memory_free = None
        self.average_accuracy_train = None
        self.average_accuracy_test = 0
        self.average_loss_train = None
        self.average_loss_test = None

    def train_model(self, stop_counter_max=3):
        start_time = time.time()
        correct = None
        train_loss = None
        average_accuracy_test_best = -1
        epoch = -1
        stop_counter = 0
        while stop_counter <= stop_counter_max:
            if epoch == 0:
                break
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
                save_all(model=self, model_save=True, rest_save=False)
                self.epochs = epoch + 1
                stop_counter = 0
            else:
                stop_counter += 1

        self.average_accuracy_test = average_accuracy_test_best
        self.needed_time = time.time() - start_time
        self.memory_total = get_memory_usage('total')
        self.memory_used = get_memory_usage('used')
        self.memory_free = self.memory_total - self.memory_used
        self.average_accuracy_train = 100 * correct
        self.average_loss_train = train_loss
        save_all(model=self, model_save=False, rest_save=True)

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
                        class_name = Cifar10Classifier.get_class_name(target_index=prediction[i].argmax(0))
                        plt.title(f'Prediction: {class_name} -> Correct!')
                        # plt.imshow(images[i].numpy()[0], cmap="summer")
                        plt.imshow(images[i].numpy()[0])
                        plt.show()
                    else:
                        class_name = Cifar10Classifier.get_class_name(target_index=prediction[i].argmax(0))
                        plt.title(f'Prediction: {class_name} -> Not correct!')
                        # plt.imshow(images[i].numpy()[0], cmap="autumn")
                        plt.imshow(images[i].numpy()[0])
                        plt.show()
                    shown += 1
                    if shown >= show:
                        break
                if shown >= show:
                    break

    @staticmethod
    def get_data(data, train=True, download=True):
        if data is None:
            data = datasets.CIFAR10(
                root="data",
                train=train,
                download=download,
                transform=ToTensor(),
            )
        return data

    def get_model(self, forward_dict=None, load=False, csv_index=0):
        path = f"models/cifar10_classifier/cifar10_classifier{csv_index}.pt"
        if load is True and os.path.isfile(path) is True:
            model = torch.load(path).to(self.device)
        else:
            model = NeuralNet(forward_dict=forward_dict, picture_size=32).to(self.device)
        return model

    @staticmethod
    def get_class_name(target_index):
        if target_index == 0:
            return 'airplane'
        elif target_index == 1:
            return 'automobile'
        elif target_index == 2:
            return 'bird'
        elif target_index == 3:
            return 'cat'
        elif target_index == 4:
            return 'deer'
        elif target_index == 5:
            return 'dog'
        elif target_index == 6:
            return 'frog'
        elif target_index == 7:
            return 'horse'
        elif target_index == 8:
            return 'ship'
        elif target_index == 9:
            return 'truck'
        else:
            return 'no class'
