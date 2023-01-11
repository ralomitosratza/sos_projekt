import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import os
import pickle
import pandas as pd
from pynvml.smi import nvidia_smi


class NeuralNet(nn.Module):
    def __init__(self, forward_dict=None, picture_size=28):
        super(NeuralNet, self).__init__()
        self.picture_size = picture_size
        last_output = 0
        self.forward_dict = forward_dict
        self.layer_list = nn.ModuleList()
        for step in forward_dict:
            if forward_dict[step]['action'] == 'layer':
                layer, self.picture_size, last_output = self.get_layer(forward_dict[step], self.picture_size,
                                                                       last_output)
                self.layer_list.append(layer)
            elif forward_dict[step]['action'] == 'f.max_pool2d':
                self.picture_size = int(self.picture_size/forward_dict[step]['kernel_size'])
        self.view_dim2 = self.picture_size*self.picture_size*last_output

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
            elif self.forward_dict[step]['action'] == 'f.softmax':
                x = f.softmax(x, dim=self.forward_dict[step]['dim'])
            elif self.forward_dict[step]['action'] == 'f.sigmoid':
                x = f.sigmoid(x)
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
                return nn.Linear(size * size * out, layer['out']), size, out
            else:
                return nn.Linear(layer['in'], layer['out']), size, out
        else:
            return None


def get_model(self, forward_dict=None, load=False, csv_index=0, picture_size=28):
    path = f'{self.model_path}{csv_index}.pt'
    if load is True and os.path.isfile(path) is True:
        model = torch.load(path).to(self.device)
    else:
        model = NeuralNet(forward_dict=forward_dict, picture_size=picture_size).to(self.device)
    return model


def save_all(model, model_save=True, rest_save=True):
    if os.path.exists(model.csv_path):
        num = len(pd.read_csv(model.csv_path))
    else:
        num = 0
    if rest_save is True:
        df = pd.DataFrame({'average_accuracy_test': model.average_accuracy_test,
                           'average_loss_test': model.average_loss_test,
                           'needed_time': model.needed_time, 'memory_used': model.memory_used/model.memory_total,
                           'memory_total': model.memory_total, 'epochs': model.epochs, 'batch_size': model.batch_size,
                           'loss_function': model.loss_fn, 'optimizer': model.optimizer, 'learning_rate': model.lr,
                           'momentum': model.momentum}, index=[1])
        df.to_csv(model.csv_path, mode='a', header=not os.path.exists(model.csv_path), index=False)
        forward_dict = open(f'{model.dict_path}{num}.pkl', 'wb')
        pickle.dump(model.forward_dict, forward_dict)
    if model_save is True:
        torch.save(model.model, f'{model.model_path}{num}.pt')


def get_loss_fn(loss_fn):
    if loss_fn is None:
        loss_fn = f.nll_loss
    return loss_fn


def get_optimizer(model, optimizer=None):
    if optimizer is None:
        optimizer = optim.SGD(model.model.parameters(), lr=model.lr, momentum=model.momentum)
    elif optimizer == 'optim.SGD':
        optimizer = optim.SGD(model.model.parameters(), lr=model.lr, momentum=model.momentum)
    elif optimizer == 'optim.Adam':
        optimizer = optim.Adam(model.model.parameters(), lr=model.lr, weight_decay=model.weight_decay)
    elif optimizer == 'optim.Adadelta':
        optimizer = optim.Adadelta(model.model.parameters())
    elif optimizer == 'optim.Adagrad':
        optimizer = optim.Adagrad(model.model.parameters())
    elif optimizer == 'optim.AdamW':
        optimizer = optim.AdamW(model.model.parameters())
    elif optimizer == 'optim.SparseAdam':
        optimizer = optim.SparseAdam(model.model.parameters())
    elif optimizer == 'optim.Adamax':
        optimizer = optim.Adamax(model.model.parameters())
    elif optimizer == 'optim.ASGD':
        optimizer = optim.ASGD(model.model.parameters())
    elif optimizer == 'optim.LBFGS':
        optimizer = optim.LBFGS(model.model.parameters())
    elif optimizer == 'optim.NAdam':
        optimizer = optim.NAdam(model.model.parameters())
    elif optimizer == 'optim.RAdam':
        optimizer = optim.RAdam(model.model.parameters())
    elif optimizer == 'optim.RMSprop':
        optimizer = optim.RMSprop(model.model.parameters())
    elif optimizer == 'optim.Rprop':
        optimizer = optim.Rprop(model.model.parameters())
    return optimizer


def get_memory_usage(option):
    return nvidia_smi.getInstance().DeviceQuery('memory.' + option)['gpu'][0]['fb_memory_usage'][option]
