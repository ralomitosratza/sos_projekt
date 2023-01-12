import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as f
import pickle
import os


def get_parameter_sets(classifier='digit_identifier'):
    forward_sets = []
    if classifier == 'digit_identifier':
        input_channels = 1
        output_channels = 10
        conv_layer_kernel_size = 3  # bis 3 conv -> sonst kernel zu klein
        pool_layer_kernel_size = 2

    elif classifier == 'catdog_classifier':
        input_channels = 3
        output_channels = 2
        conv_layer_kernel_size = 4
        pool_layer_kernel_size = 2
    elif classifier == 'cifar10_classifier':
        input_channels = 3
        output_channels = 10
        conv_layer_kernel_size = 2
        pool_layer_kernel_size = 2
    else:
        input_channels = 0
        output_channels = 0
        conv_layer_kernel_size = 0
        pool_layer_kernel_size = 0

    if classifier == '':
        # architecture one  --------- 2 conv, dropout, 2 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step9_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s9o in step9_out:
                    dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                     'kernel_size': conv_layer_kernel_size},
                           'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                           'step3': {'action': 'f.relu'},
                           'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                     'kernel_size': conv_layer_kernel_size},
                           'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
                           'step6': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                           'step7': {'action': 'f.relu'},
                           'step8': {'action': 'view', 'dim1': -1},
                           'step9': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s9o},
                           'step10': {'action': 'f.relu'},
                           'step11': {'action': 'layer', 'layer': 'linear', 'in': s9o, 'out': output_channels},
                           'step12': {'action': 'f.log_softmax', 'dim': 1}}
                    forward_sets.append(dic)

    if classifier == '':
        # architecture two --------- 3 conv, dropout, 2 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step12_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s12o in step12_out:
                        dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                         'kernel_size': conv_layer_kernel_size},
                               'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                               'step3': {'action': 'f.relu'},
                               'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                         'kernel_size': conv_layer_kernel_size},
                               'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                               'step6': {'action': 'f.relu'},
                               'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                         'kernel_size': conv_layer_kernel_size},
                               'step8': {'action': 'layer', 'layer': 'conv_dropout2d'},
                               'step9': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                               'step10': {'action': 'f.relu'},
                               'step11': {'action': 'view', 'dim1': -1},
                               'step12': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s12o},
                               'step13': {'action': 'f.relu'},
                               'step14': {'action': 'layer', 'layer': 'linear', 'in': s12o, 'out': output_channels},
                               'step15': {'action': 'f.log_softmax', 'dim': 1}}
                        forward_sets.append(dic)

    if classifier == '':
        # architecture three --------- 2 conv, dropout, 3 linear
        step1_out = [100]  # 5, 20 ---> digit 50, 100
        step4_out = [200]  # 20, 70 ---> digit 100, 200
        step9_out = [4000]  # 1000, 3000 ---> digit 2500, 4000
        step11_out = [2000]  # 500, 1000 ---> digit 1000, 2000
        for s1o in step1_out:
            for s4o in step4_out:
                for s9o in step9_out:
                    for s11o in step11_out:
                        dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                         'kernel_size': conv_layer_kernel_size},
                               'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                               'step3': {'action': 'f.relu'},
                               'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                         'kernel_size': conv_layer_kernel_size},
                               'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
                               'step6': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                               'step7': {'action': 'f.relu'},
                               'step8': {'action': 'view', 'dim1': -1},
                               'step9': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s9o},
                               'step10': {'action': 'f.relu'},
                               'step11': {'action': 'layer', 'layer': 'linear', 'in': s9o, 'out': s11o},
                               'step12': {'action': 'f.relu'},
                               'step13': {'action': 'layer', 'layer': 'linear', 'in': s11o, 'out': output_channels},
                               'step14': {'action': 'f.log_softmax', 'dim': 1}}
                        forward_sets.append(dic)

    if classifier == 'digit_identifier':
        i = 0
        # architecture four ---------- 3 conv, dropout, 3 linear
        step1_out = [200, 700]  # 5, 20, 200, 700
        step4_out = [1000, 1500]  # 20, 70, 1000, 1500
        step7_out = [1500, 3000]  # 70, 100, 1500, 3000
        step12_out = [7000, 10000]  # 1000, 3000, 7000, 10000
        step14_out = [2500, 5000]  # 500, 1000, 2500, 5000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s12o in step12_out:
                        for s14o in step14_out:
                            i += 1
                            if i > 29:
                                dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step3': {'action': 'f.relu'},
                                       'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step6': {'action': 'f.relu'},
                                       'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step8': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                       'step9': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step10': {'action': 'f.relu'},
                                       'step11': {'action': 'view', 'dim1': -1},
                                       'step12': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s12o},
                                       'step13': {'action': 'f.relu'},
                                       'step14': {'action': 'layer', 'layer': 'linear', 'in': s12o, 'out': s14o},
                                       'step15': {'action': 'f.relu'},
                                       'step16': {'action': 'layer', 'layer': 'linear', 'in': s14o,
                                                  'out': output_channels},
                                       'step17': {'action': 'f.log_softmax', 'dim': 1}}
                                forward_sets.append(dic)

    if classifier == '':
        # architecture five -------- 4 conv, dropout, 2 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step10_out = [100, 200]  # 100, 200
        step15_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s10o in step10_out:
                        for s15o in step15_out:
                            dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                             'kernel_size': conv_layer_kernel_size},
                                   'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                   'step3': {'action': 'f.relu'},
                                   'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                             'kernel_size': conv_layer_kernel_size},
                                   'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                   'step6': {'action': 'f.relu'},
                                   'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                             'kernel_size': conv_layer_kernel_size},
                                   'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                   'step9': {'action': 'f.relu'},
                                   'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
                                              'kernel_size': conv_layer_kernel_size},
                                   'step11': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                   'step12': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                   'step13': {'action': 'f.relu'},
                                   'step14': {'action': 'view', 'dim1': -1},
                                   'step15': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s15o},
                                   'step16': {'action': 'f.relu'},
                                   'step17': {'action': 'layer', 'layer': 'linear', 'in': s15o, 'out': output_channels},
                                   'step18': {'action': 'f.log_softmax', 'dim': 1}}
                            forward_sets.append(dic)

    if classifier == '':
        # architecture six -------- 4 conv, dropout, 3 linear
        step1_out = [30, 50]  # 5, 20, 30, 50
        step4_out = [100, 120]  # 20, 70, 100, 120
        step7_out = [150, 200]  # 70, 100, 150, 200
        step10_out = [300, 350]  # 100, 200, 300, 350
        step15_out = [3500, 4000]  # 1000, 3000, 3500, 4000
        step17_out = [1500, 2000]  # 500, 1000, 1500, 2000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s10o in step10_out:
                        for s15o in step15_out:
                            for s17o in step17_out:
                                dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step3': {'action': 'f.relu'},
                                       'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step6': {'action': 'f.relu'},
                                       'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step9': {'action': 'f.relu'},
                                       'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
                                                  'kernel_size': conv_layer_kernel_size},
                                       'step11': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                       'step12': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step13': {'action': 'f.relu'},
                                       'step14': {'action': 'view', 'dim1': -1},
                                       'step15': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s15o},
                                       'step16': {'action': 'f.relu'},
                                       'step17': {'action': 'layer', 'layer': 'linear', 'in': s15o, 'out': s17o},
                                       'step18': {'action': 'f.relu'},
                                       'step19': {'action': 'layer', 'layer': 'linear', 'in': s17o,
                                                  'out': output_channels},
                                       'step20': {'action': 'f.log_softmax', 'dim': 1}}
                                forward_sets.append(dic)

    if classifier == '':
        # architecture seven -------- 5 conv, dropout, 2 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step10_out = [100, 200]  # 100, 200
        step13_out = [200, 300]  # 200, 300
        step18_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s10o in step10_out:
                        for s13o in step13_out:
                            for s18o in step18_out:
                                dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step3': {'action': 'f.relu'},
                                       'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step6': {'action': 'f.relu'},
                                       'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                                 'kernel_size': conv_layer_kernel_size},
                                       'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step9': {'action': 'f.relu'},
                                       'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
                                                  'kernel_size': conv_layer_kernel_size},
                                       'step11': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step12': {'action': 'f.relu'},
                                       'step13': {'action': 'layer', 'layer': 'conv2d', 'in': s10o, 'out': s13o,
                                                  'kernel_size': conv_layer_kernel_size},
                                       'step14': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                       'step15': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                       'step16': {'action': 'f.relu'},
                                       'step17': {'action': 'view', 'dim1': -1},
                                       'step18': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s18o},
                                       'step19': {'action': 'f.relu'},
                                       'step20': {'action': 'layer', 'layer': 'linear', 'in': s18o,
                                                  'out': output_channels},
                                       'step21': {'action': 'f.log_softmax', 'dim': 1}}
                                forward_sets.append(dic)

    if classifier == '':
        # architecture eight -------- 5 conv, dropout, 3 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step10_out = [100, 200]  # 100, 200
        step13_out = [200, 300]  # 200, 300
        step18_out = [1000, 3000]  # 1000, 3000
        step20_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s10o in step10_out:
                        for s13o in step13_out:
                            for s18o in step18_out:
                                for s20o in step20_out:
                                    dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels,
                                                     'out': s1o,
                                                     'kernel_size': conv_layer_kernel_size},
                                           'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                           'step3': {'action': 'f.relu'},
                                           'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
                                                     'kernel_size': conv_layer_kernel_size},
                                           'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                           'step6': {'action': 'f.relu'},
                                           'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
                                                     'kernel_size': conv_layer_kernel_size},
                                           'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                           'step9': {'action': 'f.relu'},
                                           'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
                                                      'kernel_size': conv_layer_kernel_size},
                                           'step11': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                           'step12': {'action': 'f.relu'},
                                           'step13': {'action': 'layer', 'layer': 'conv2d', 'in': s10o, 'out': s13o,
                                                      'kernel_size': conv_layer_kernel_size},
                                           'step14': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                           'step15': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
                                           'step16': {'action': 'f.relu'},
                                           'step17': {'action': 'view', 'dim1': -1},
                                           'step18': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s18o},
                                           'step19': {'action': 'f.relu'},
                                           'step20': {'action': 'layer', 'layer': 'linear', 'in': s18o, 'out': s20o},
                                           'step21': {'action': 'f.relu'},
                                           'step22': {'action': 'layer', 'layer': 'linear', 'in': s20o,
                                                      'out': output_channels},
                                           'step23': {'action': 'f.log_softmax', 'dim': 1}}
                                    forward_sets.append(dic)

    batch_size = [64]
    optimizer = ['optim.SGD']
    sgd_lr = [0.1]
    sgd_momentum = [0.5]
    loss_fn = [f.nll_loss]

    para_sets = []

    for opt in optimizer:
        for lfn in loss_fn:
            for bs in batch_size:
                if opt == 'optim.SGD':
                    for lr in sgd_lr:
                        for momentum in sgd_momentum:
                            dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': lr,
                                   'momentum': momentum,
                                   'weight_decay': 0}
                            para_sets.append(dic)
                elif opt == 'optim.Adadelta':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.Adagrad':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.AdamW':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.SparseAdam':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.Adamax':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.ASGD':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.LBFGS':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.NAdam':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.RAdam':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.RMSprop':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.Rprop':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0, 'weight_decay': 0,
                           'momentum': 0}
                    para_sets.append(dic)
                elif opt == 'optim.Adam':
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0.1, 'momentum': 0.8,
                           'weight_decay': 0}
                    para_sets.append(dic)
                else:
                    print('optimizer not treated')
                    dic = {'batch_size': bs, 'loss_fn': lfn, 'optimizer': opt, 'lr': 0.1, 'momentum': 0.8,
                           'weight_decay': 0}
                    para_sets.append(dic)

    return forward_sets, para_sets


def plot_pandas(classifier='digit_identifier', start_index=0, end_index=None, architecture=False):
    if classifier == 'digit_identifier':
        path = 'panda_tables/runs_digit_identifier.csv'
        x = ['one', 'two', 'three', 'four']
        x_ticks = [38, 46, 62, 78]
    elif classifier == 'catdog_classifier':
        path = 'panda_tables/runs_catdog_classifier.csv'
        x = ['one', 'two', 'four', 'six', 'eight', 'six']
        x_ticks = [0, 8, 24, 56, 120, 248]
    elif classifier == 'cifar10_classifier':
        path = 'panda_tables/runs_cifar10_classifier.csv'
        x = ['one', 'two', 'four', 'six']
        x_ticks = [0, 8, 24, 56]
    else:
        path = ''
        x = []
        x_ticks = []
    df = pd.read_csv(path)
    if end_index is None:
        end_index = len(pd.read_csv(path))

    list_of_indices = list(range(start_index, end_index, 1))
    df = df.iloc[list_of_indices, :]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(right=0.75)
    plt.title(classifier)
    ax.plot(df['average_accuracy_test'], color='steelblue')
    if architecture is True:
        plt.xticks(x_ticks, x)
        ax.set_xlabel('Architecture')
    else:
        ax.set_xlabel('Run')
    ax.set_ylabel('average accuracy test', color='steelblue')
    if classifier == 'digit_identifier':
        ax.axis(ymin=98, ymax=100)
    ax2 = ax.twinx()
    ax2.plot(df['memory_used'], color='red')
    ax2.set_ylabel('memory used', color='red')
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.07))
    ax3.plot(df['needed_time'], color='green')
    ax3.set_ylabel('needed time', color='green')
    if architecture is True:
        plt.savefig(f'pics/{classifier}_architecture.png', bbox_inches='tight' )
    else:
        plt.savefig(f'pics/{classifier}_runs.png', bbox_inches='tight')
    plt.show()


def show_set(classifier='digit_identifier', csv_index=0):
    if classifier == 'digit_identifier':
        path = 'dictionarys/digit_identifier/forward_dictionary'
        path_csv = 'panda_tables/runs_digit_identifier.csv'
    elif classifier == 'catdog_classifier':
        path = 'dictionarys/catdog_classifier/forward_dictionary'
        path_csv = 'panda_tables/runs_catdog_classifier.csv'
    elif classifier == 'cifar10_classifier':
        path = 'dictionarys/cifar10_classifier/forward_dictionary'
        path_csv = 'panda_tables/runs_cifar10_classifier.csv'
    else:
        path = ''
        path_csv = ''

    df = pd.read_csv(path_csv)
    if csv_index == 'highest_accuracy':
        csv_index = df['average_accuracy_test'].idxmax()

    with open(f'{path}{csv_index}.pkl', 'rb') as handle:
        dictionary = pickle.load(handle)
        print(f'Classifier: {classifier},     Index: {csv_index}')
        print(f'LAYER + LAYERSIZE')
        for dic in dictionary:
            print(dictionary[dic])
        print(f'\nOTHER PARAMETERS')
        print(df.iloc[csv_index])
        print('\n\n')
    return csv_index


def save_means_of_csv_to_csv(classifier='digit_identifier', architecture='one', start_index=0, end_index=0):
    if classifier == 'digit_identifier':
        path_csv = 'panda_tables/runs_digit_identifier.csv'
        path_mean = 'panda_tables/mean_digit_identifier.csv'
    elif classifier == 'catdog_classifier':
        path_csv = 'panda_tables/runs_catdog_classifier.csv'
        path_mean = 'panda_tables/mean_catdog_classifier.csv'
    elif classifier == 'cifar10_classifier':
        path_csv = 'panda_tables/runs_cifar10_classifier.csv'
        path_mean = 'panda_tables/mean_cifar10_classifier.csv'
    else:
        path_csv = ''
        path_mean = ''

    df = pd.read_csv(path_csv)
    specific_rows = df.iloc[start_index:end_index]
    mean_average_accuracy_test = specific_rows['average_accuracy_test'].mean()
    mean_needed_time = specific_rows['needed_time'].mean()
    mean_memory_used = specific_rows['memory_used'].mean()
    df = pd.DataFrame({'architecture': architecture, 'mean_average_accuracy_test': mean_average_accuracy_test,
                       'mean_needed_time': mean_needed_time, 'mean_memory_used': mean_memory_used}, index=[1])
    df.to_csv(path_mean, mode='a', header=not os.path.exists(path_mean), index=False)


def show_means(classifier='digit_identifier'):
    if classifier == 'digit_identifier':
        path = 'panda_tables/mean_digit_identifier.csv'
    elif classifier == 'catdog_classifier':
        path = 'panda_tables/mean_catdog_classifier.csv'
    elif classifier == 'cifar10_classifier':
        path = 'panda_tables/mean_cifar10_classifier.csv'
    else:
        path = ''

    df = pd.read_csv(path)
    x = list(df.loc[:, 'architecture'].values)
    plt.rcParams['figure.autolayout'] = True
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(right=0.75)
    plt.title(classifier)
    x_ticks = range(len(x))
    plt.xticks(x_ticks, x)
    ax.plot(df['mean_average_accuracy_test'], color='steelblue')
    ax.set_xlabel('Architecture')
    ax.set_ylabel('mean_average accuracy test', color='steelblue')
    ax2 = ax.twinx()
    ax2.plot(df['mean_memory_used'], color='red')
    ax2.set_ylabel('mean_memory used', color='red')
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.07))
    ax3.plot(df['mean_needed_time'], color='green')
    ax3.set_ylabel('mean_needed time', color='green')
    plt.savefig(f'pics/{classifier}_mean.png', bbox_inches='tight')
    plt.show()
