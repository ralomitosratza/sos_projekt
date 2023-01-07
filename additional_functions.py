import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as f
import pickle


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

    if classifier == 'cifar10_classifier':
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

    if classifier == 'cifar10_classifier':
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

    if classifier == 'cifar10_classifier':
        # architecture three --------- 2 conv, dropout, 3 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step9_out = [1000, 3000]  # 1000, 3000
        step11_out = [500, 1000]  # 500, 1000
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

    if classifier == 'cifar10_classifier':
        # architecture four ---------- 3 conv, dropout, 3 linear
        step1_out = [20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step12_out = [1000, 3000]  # 1000, 3000
        step14_out = [500, 1000]  # 500, 1000
        for s1o in step1_out:
            for s4o in step4_out:
                for s7o in step7_out:
                    for s12o in step12_out:
                        for s14o in step14_out:
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
                                   'step16': {'action': 'layer', 'layer': 'linear', 'in': s14o, 'out': output_channels},
                                   'step17': {'action': 'f.log_softmax', 'dim': 1}}
                            forward_sets.append(dic)

    if classifier == 'cifar10_classifier':
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

    if classifier == 'cifar10_classifier':
        # architecture six -------- 4 conv, dropout, 3 linear
        step1_out = [5, 20]  # 5, 20
        step4_out = [20, 70]  # 20, 70
        step7_out = [70, 100]  # 70, 100
        step10_out = [100, 200]  # 100, 200
        step15_out = [1000, 3000]  # 1000, 3000
        step17_out = [500, 1000]  # 500, 1000
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

    if classifier == 'catdog_classifier':
        i = 0
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
                                    i += 1
                                    if i <= 11:
                                        continue
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

    # if classifier == 'digit_identifier' or classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture nine  --------- 2 conv, 2 linear
    #     step1_out = [5, 20]  # 5, 20
    #     step4_out = [20, 70]  # 20, 70
    #     step9_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s9o in step9_out:
    #                 dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                  'kernel_size': conv_layer_kernel_size},
    #                        'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                        'step3': {'action': 'f.relu'},
    #                        'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                  'kernel_size': conv_layer_kernel_size},
    #                        'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                        'step6': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                        'step7': {'action': 'f.relu'},
    #                        'step8': {'action': 'view', 'dim1': -1},
    #                        'step9': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s9o},
    #                        'step10': {'action': 'f.relu'},
    #                        'step11': {'action': 'layer', 'layer': 'linear', 'in': s9o, 'out': output_channels},
    #                        'step12': {'action': 'f.log_softmax', 'dim': 1}}
    #                 forward_sets.append(dic)
    #
    # if classifier == 'digit_identifier' or classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture ten --------- 3 conv, 2 linear
    #     step1_out = [5, 20]  # 5, 20
    #     step4_out = [20, 70]  # 20, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step12_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s12o in step12_out:
    #                     dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                      'kernel_size': conv_layer_kernel_size},
    #                            'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                            'step3': {'action': 'f.relu'},
    #                            'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                      'kernel_size': conv_layer_kernel_size},
    #                            'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                            'step6': {'action': 'f.relu'},
    #                            'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                      'kernel_size': conv_layer_kernel_size},
    #                            'step8': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                            'step9': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                            'step10': {'action': 'f.relu'},
    #                            'step11': {'action': 'view', 'dim1': -1},
    #                            'step12': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s12o},
    #                            'step13': {'action': 'f.relu'},
    #                            'step14': {'action': 'layer', 'layer': 'linear', 'in': s12o, 'out': output_channels},
    #                            'step15': {'action': 'f.log_softmax', 'dim': 1}}
    #                     forward_sets.append(dic)
    #
    # if classifier == 'digit_identifier' or classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture eleven --------- 2 conv, 3 linear
    #     step1_out = [5, 20]  # 5, 20
    #     step4_out = [20, 70]  # 20, 70
    #     step9_out = [1000, 3000]  # 1000, 3000
    #     step11_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s9o in step9_out:
    #                 for s11o in step11_out:
    #                     dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                      'kernel_size': conv_layer_kernel_size},
    #                            'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                            'step3': {'action': 'f.relu'},
    #                            'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                      'kernel_size': conv_layer_kernel_size},
    #                            'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                            'step6': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                            'step7': {'action': 'f.relu'},
    #                            'step8': {'action': 'view', 'dim1': -1},
    #                            'step9': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s9o},
    #                            'step10': {'action': 'f.relu'},
    #                            'step11': {'action': 'layer', 'layer': 'linear', 'in': s9o, 'out': s11o},
    #                            'step12': {'action': 'f.relu'},
    #                            'step13': {'action': 'layer', 'layer': 'linear', 'in': s11o, 'out': output_channels},
    #                            'step14': {'action': 'f.log_softmax', 'dim': 1}}
    #                     forward_sets.append(dic)
    #
    # if classifier == 'digit_identifier' or classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture twelve ---------- 3 conv, 3 linear
    #     step1_out = [5, 20]  # 5, 10, 20
    #     step4_out = [20, 70]  # 20, 50, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step12_out = [1000, 3000]  # 1000, 3000
    #     step14_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s12o in step12_out:
    #                     for s14o in step14_out:
    #                         dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step3': {'action': 'f.relu'},
    #                                'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step6': {'action': 'f.relu'},
    #                                'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step8': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                                'step9': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step10': {'action': 'f.relu'},
    #                                'step11': {'action': 'view', 'dim1': -1},
    #                                'step12': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s12o},
    #                                'step13': {'action': 'f.relu'},
    #                                'step14': {'action': 'layer', 'layer': 'linear', 'in': s12o, 'out': s14o},
    #                                'step15': {'action': 'f.relu'},
    #                                'step16': {'action': 'layer', 'layer': 'linear', 'in': s14o, 'out': output_channels},
    #                                'step17': {'action': 'f.log_softmax', 'dim': 1}}
    #                         forward_sets.append(dic)
    #
    # if classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture thirteen -------- 4 conv, 2 linear
    #     step1_out = [5, 20]  # 5, 20
    #     step4_out = [20, 70]  # 20, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step10_out = [100, 200]  # 100, 200
    #     step15_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s10o in step10_out:
    #                     for s15o in step15_out:
    #                         dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step3': {'action': 'f.relu'},
    #                                'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step6': {'action': 'f.relu'},
    #                                'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                          'kernel_size': conv_layer_kernel_size},
    #                                'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step9': {'action': 'f.relu'},
    #                                'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
    #                                           'kernel_size': conv_layer_kernel_size},
    #                                'step11': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                                'step12': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                'step13': {'action': 'f.relu'},
    #                                'step14': {'action': 'view', 'dim1': -1},
    #                                'step15': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s15o},
    #                                'step16': {'action': 'f.relu'},
    #                                'step17': {'action': 'layer', 'layer': 'linear', 'in': s15o, 'out': output_channels},
    #                                'step18': {'action': 'f.log_softmax', 'dim': 1}}
    #                         forward_sets.append(dic)
    #
    # if classifier == 'catdog_classifier' or classifier == 'cifar10_classifier':
    #     # architecture fourteen -------- 4 conv, 3 linear
    #     step1_out = [5, 20]  # 5, 10, 20
    #     step4_out = [20, 70]  # 20, 50, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step10_out = [100, 200]  # 100, 200
    #     step15_out = [1000, 3000]  # 1000, 3000
    #     step17_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s10o in step10_out:
    #                     for s15o in step15_out:
    #                         for s17o in step17_out:
    #                             dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step3': {'action': 'f.relu'},
    #                                    'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step6': {'action': 'f.relu'},
    #                                    'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step9': {'action': 'f.relu'},
    #                                    'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
    #                                               'kernel_size': conv_layer_kernel_size},
    #                                    'step11': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                                    'step12': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step13': {'action': 'f.relu'},
    #                                    'step14': {'action': 'view', 'dim1': -1},
    #                                    'step15': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s15o},
    #                                    'step16': {'action': 'f.relu'},
    #                                    'step17': {'action': 'layer', 'layer': 'linear', 'in': s15o, 'out': s17o},
    #                                    'step18': {'action': 'f.relu'},
    #                                    'step19': {'action': 'layer', 'layer': 'linear', 'in': s17o,
    #                                               'out': output_channels},
    #                                    'step20': {'action': 'f.log_softmax', 'dim': 1}}
    #                             forward_sets.append(dic)
    #
    # if classifier == 'catdog_classifier':
    #     # architecture fifteen -------- 5 conv, 2 linear
    #     step1_out = [5, 20]  # 5, 10, 20
    #     step4_out = [20, 70]  # 20, 50, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step10_out = [100, 200]  # 100, 200
    #     step13_out = [200, 300]  # 200, 300
    #     step18_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s10o in step10_out:
    #                     for s13o in step13_out:
    #                         for s18o in step18_out:
    #                             dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels, 'out': s1o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step3': {'action': 'f.relu'},
    #                                    'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step6': {'action': 'f.relu'},
    #                                    'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                              'kernel_size': conv_layer_kernel_size},
    #                                    'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step9': {'action': 'f.relu'},
    #                                    'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
    #                                               'kernel_size': conv_layer_kernel_size},
    #                                    'step11': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step12': {'action': 'f.relu'},
    #                                    'step13': {'action': 'layer', 'layer': 'conv2d', 'in': s10o, 'out': s13o,
    #                                               'kernel_size': conv_layer_kernel_size},
    #                                    'step14': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                                    'step15': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                    'step16': {'action': 'f.relu'},
    #                                    'step17': {'action': 'view', 'dim1': -1},
    #                                    'step18': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s18o},
    #                                    'step19': {'action': 'f.relu'},
    #                                    'step20': {'action': 'layer', 'layer': 'linear', 'in': s18o,
    #                                               'out': output_channels},
    #                                    'step21': {'action': 'f.log_softmax', 'dim': 1}}
    #                             forward_sets.append(dic)
    #
    # if classifier == 'catdog_classifier':
    #     # architecture sixteen -------- 5 conv, 3 linear
    #     step1_out = [5, 20]  # 5, 10, 20
    #     step4_out = [20, 70]  # 20, 50, 70
    #     step7_out = [70, 100]  # 70, 100
    #     step10_out = [100, 200]  # 100, 200
    #     step13_out = [200, 300]  # 200, 300
    #     step18_out = [1000, 3000]  # 1000, 3000
    #     step20_out = [500, 1000]  # 500, 1000
    #     for s1o in step1_out:
    #         for s4o in step4_out:
    #             for s7o in step7_out:
    #                 for s10o in step10_out:
    #                     for s13o in step13_out:
    #                         for s18o in step18_out:
    #                             for s20o in step20_out:
    #                                 dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': input_channels,
    #                                                  'out': s1o,
    #                                                  'kernel_size': conv_layer_kernel_size},
    #                                        'step2': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                        'step3': {'action': 'f.relu'},
    #                                        'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o, 'out': s4o,
    #                                                  'kernel_size': conv_layer_kernel_size},
    #                                        'step5': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                        'step6': {'action': 'f.relu'},
    #                                        'step7': {'action': 'layer', 'layer': 'conv2d', 'in': s4o, 'out': s7o,
    #                                                  'kernel_size': conv_layer_kernel_size},
    #                                        'step8': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                        'step9': {'action': 'f.relu'},
    #                                        'step10': {'action': 'layer', 'layer': 'conv2d', 'in': s7o, 'out': s10o,
    #                                                   'kernel_size': conv_layer_kernel_size},
    #                                        'step11': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                        'step12': {'action': 'f.relu'},
    #                                        'step13': {'action': 'layer', 'layer': 'conv2d', 'in': s10o, 'out': s13o,
    #                                                   'kernel_size': conv_layer_kernel_size},
    #                                        'step14': {'action': 'layer', 'layer': 'conv_dropout2d'},
    #                                        'step15': {'action': 'f.max_pool2d', 'kernel_size': pool_layer_kernel_size},
    #                                        'step16': {'action': 'f.relu'},
    #                                        'step17': {'action': 'view', 'dim1': -1},
    #                                        'step18': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s18o},
    #                                        'step19': {'action': 'f.relu'},
    #                                        'step20': {'action': 'layer', 'layer': 'linear', 'in': s18o, 'out': s20o},
    #                                        'step21': {'action': 'f.relu'},
    #                                        'step22': {'action': 'layer', 'layer': 'linear', 'in': s20o,
    #                                                   'out': output_channels},
    #                                        'step23': {'action': 'f.log_softmax', 'dim': 1}}
    #                                 forward_sets.append(dic)

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


def plot_pandas(classifier='digit_identifier', start_index=0, end_index=None):
    if classifier == 'digit_identifier':
        path = 'panda_tables/runs_digit_identifier.csv'
    elif classifier == 'catdog_classifier':
        path = 'panda_tables/runs_catdog_classifier.csv'
    elif classifier == 'cifar10_classifier':
        path = 'panda_tables/runs_cifar10_classifier.csv'
    else:
        path = ''
    df = pd.read_csv(path)
    if end_index is None:
        end_index = len(pd.read_csv(path))

    list_of_indices = list(range(start_index, end_index, 1))
    df = df.iloc[list_of_indices, :]
    fig, ax = plt.subplots(figsize=(15, 7))
    fig.subplots_adjust(right=0.75)
    ax.plot(df['average_accuracy_test'], color='steelblue')
    ax.set_xlabel('Run')
    ax.set_ylabel('average accuracy test', color='steelblue')
    if classifier == 'digit_identifier':
        ax.axis(ymin=98, ymax=100)
    ax2 = ax.twinx()
    ax2.plot(df['memory_used'], color='red')
    ax2.set_ylabel('memory used', color='red')
    ax3 = ax.twinx()
    ax3.spines.right.set_position(("axes", 1.2))
    ax3.plot(df['needed_time'], color='green')
    ax3.set_ylabel('needed time', color='green')
    plt.show()


def show_set(classifier='digit_identifier', csv_index=0):
    if classifier == 'digit_identifier':
        path = 'dictionarys/digit_identifier/forward_dictionary'
    elif classifier == 'catdog_classifier':
        path = 'dictionarys/catdog_classifier/forward_dictionary'
    elif classifier == 'cifar10_classifier':
        path = 'dictionarys/cifar10_classifier/forward_dictionary'
    else:
        path = ''
    with open(f'{path}{csv_index}.pkl', 'rb') as handle:
        dictionary = pickle.load(handle)
        for dic in dictionary:
            print(dictionary[dic])
