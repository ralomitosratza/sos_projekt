import torch.nn as nn
import torch.nn.functional as f
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
                layer, self.picture_size, last_output = self.get_layer(forward_dict[step], self.picture_size, last_output)
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


def get_loss_fn(loss_fn):
    if loss_fn is None:
        loss_fn = f.nll_loss
    return loss_fn


def get_memory_usage(option):
    return nvidia_smi.getInstance().DeviceQuery('memory.' + option)['gpu'][0]['fb_memory_usage'][option]
