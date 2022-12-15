from digit_identifier import DigitIdentifier

# TODO: define hyper parameters and stuff
epochs = 1
forward_dict = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': 1, 'out': 10, 'kernel_size': 5},
                'step2': {'action': 'f.max_pool2d', 'kernel_size': 2},
                'step3': {'action': 'f.relu'},
                'step4': {'action': 'layer', 'layer': 'conv2d', 'in': 10, 'out': 20, 'kernel_size': 5},
                'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
                'step6': {'action': 'f.max_pool2d', 'kernel_size': 2},
                'step7': {'action': 'f.relu'},
                'step8': {'action': 'view', 'dim1': -1, 'dim2': 320},
                'step9': {'action': 'layer', 'layer': 'linear', 'in': 320, 'out': 60},
                'step10': {'action': 'f.relu'},
                'step11': {'action': 'layer', 'layer': 'linear', 'in': 60, 'out': 10},
                'step12': {'action': 'f.log_softmax', 'dim': 1}}

# TODO: digit_identifier loop
DI = DigitIdentifier(forward_dict=forward_dict)
DI.train_model(epochs)
# print(DI.needed_time)
# print(DI.memory_total)
# print(DI.memory_used)
# print(DI.memory_free)
# DI.test_model()
# DI.show_results(show=5)
# TODO: handle return values

