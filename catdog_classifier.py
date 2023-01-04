import torch
import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
import time
import random
from model_functions import NeuralNet, get_loss_fn, get_memory_usage
from torchvision import transforms
from PIL import Image
from os import listdir


class CatdogClassifier:
    def __init__(self, batch_size=64, load=False, csv_index=0, forward_dict=None, loss_fn=None, optimizer=None, lr=0.1,
                 momentum=0.8, weight_decay=0.0001, info=False):
        self.average_accuracy_test = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.info = info
        if self.info:
            print(f"Using {self.device} device")
        self.epochs = None
        self.batch_size = batch_size
        if not load:
            self.train_data = self.get_data(train=True, batch_size=self.batch_size)
        self.test_data = self.get_data(train=False, batch_size=self.batch_size)
        self.forward_dict = forward_dict
        self.model = self.get_model(forward_dict=forward_dict, load=load, csv_index=csv_index)
        self.best_model = None
        if load is False:
            self.loss_fn = get_loss_fn(loss_fn)
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

    def get_optimizer(self, optimizer):
        if optimizer is None:
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer == 'optim.SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif optimizer == 'optim.Adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
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
            if epoch == 0:
                break
            epoch += 1
            self.model.train()
            train_loss, correct = 0, 0
            for batch_id, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                prediction = self.model(data)

                loss = self.loss_fn(prediction, target)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
                if (batch_id == 35 or batch_id == 70 or batch_id == 105 or batch_id == 140 or batch_id == 175) \
                        and self.info:
                    print(f"current loss: {loss.item()} ")
            train_loss /= len(self.train_data)
            correct /= (len(self.train_data)*self.batch_size)
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
        self.memory_total = get_memory_usage('total')
        self.memory_used = get_memory_usage('used')
        self.memory_free = self.memory_total - self.memory_used
        self.average_accuracy_train = 100 * correct
        self.average_loss_train = train_loss
        self.save_all(model=False, rest=True)

    def get_model(self, forward_dict=None, load=False, csv_index=0):
        path = f"models/catdog_classifier/catdog_classifier{csv_index}.pt"
        if load is True and os.path.isfile(path) is True:
            model = torch.load(path).to(self.device)
        else:
            model = NeuralNet(forward_dict=forward_dict, picture_size=128).to(self.device)
        return model

    @staticmethod
    def get_data(train=True, batch_size=64):
        if os.path.isfile('data/catdog/tensor_train/tensor_train.pt') and train is True:
            data = torch.load('data/catdog/tensor_train/tensor_train.pt')
        elif os.path.isfile('data/catdog/tensor_test/tensor_test.pt') and train is False:
            data = torch.load('data/catdog/tensor_test/tensor_test.pt')
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform = transforms.Compose(
                [transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor(), normalize])

            data_list = []
            target_list = []
            train_data = []
            test_data = []
            train_list = listdir('data/catdog/train/')
            random.shuffle(train_list)
            for i, tl in enumerate(train_list):
                img = Image.open('data/catdog/train/' + tl)
                img_tensor = transform(img)
                data_list.append(img_tensor)
                target = 1 if 'cat' in tl else 0

                target_list.append(target)
                if len(data_list) >= batch_size:
                    if i <= 0.25*len(train_list):
                        test_data.append((torch.stack(data_list), torch.tensor(target_list)))
                        data_list = []
                        target_list = []
                    else:
                        train_data.append((torch.stack(data_list), torch.tensor(target_list)))
                        data_list = []
                        target_list = []
            torch.save(train_data, 'data/catdog/tensor_train/tensor_train.pt')
            torch.save(test_data, 'data/catdog/tensor_test/tensor_test.pt')
            data = CatdogClassifier.get_data(train=train, batch_size=batch_size)
        return data

    def save_all(self, model=True, rest=True):
        path = 'panda_tables/runs_catdog_classifier.csv'
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
            forward_dict = open(f'dictionarys/catdog_classifier/forward_dictionary{num}.pkl', 'wb')
            pickle.dump(self.forward_dict, forward_dict)
        if model is True:
            torch.save(self.model, f"models/catdog_classifier/catdog_classifier{num}.pt")

    def test_model(self):
        self.model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                test_loss += self.loss_fn(prediction, target).item()
                correct += (prediction.argmax(1) == target).type(torch.float).sum().item()
        test_loss /= len(self.test_data)
        correct /= (len(self.test_data) * self.batch_size)
        if self.info:
            print(f"Test Error: Accuracy: {(100 * correct)}%, Average loss: {test_loss} \n")
        self.average_accuracy_test = 100 * correct
        self.average_loss_test = test_loss

    def try_model(self, show=5):
        shown = 0
        self.model.eval()
        images, labels = next(iter(self.test_data))
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                for i in range(data.size()[0]):
                    if prediction[i].argmax(0) == target[i]:
                        catdog = 'Cat' if target[i] == 1 else 'Dog'
                        plt.title(f'Prediction: {catdog} -> Correct!')
                        plt.imshow(images[i].numpy()[0], cmap="summer")
                        plt.show()
                    else:
                        catdog = 'Cat' if target[i] == 1 else 'Dog'
                        plt.title(f'Prediction: {catdog} -> Not correct!')
                        plt.imshow(images[i].numpy()[0], cmap="autumn")
                        plt.show()
                    shown += 1
                    if shown >= show:
                        break
                if shown >= show:
                    break
