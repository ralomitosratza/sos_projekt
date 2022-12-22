def get_forward_set():
    step1_out = range(20, 50, 10)
    step1_kernel_size = range(2, 6, 2)
    step2_kernel_size = range(2, 3, 1)
    step4_out = range(20, 50, 20)
    step4_kernel_size = range(5, 6, 2)
    step6_kernel_size = range(2, 3, 1)
    step9_out = range(40, 90, 20)

    parameter_sets = []
    for s1o in step1_out:
        for s1ks in step1_kernel_size:
            for s2ks in step2_kernel_size:
                for s4o in step4_out:
                    for s4ks in step4_kernel_size:
                        for s6ks in step6_kernel_size:
                            for s9o in step9_out:
                                dic = {'step1': {'action': 'layer', 'layer': 'conv2d', 'in': 1, 'out': s1o,
                                                 'kernel_size': s1ks},
                                       'step2': {'action': 'f.max_pool2d', 'kernel_size': s2ks},
                                       'step3': {'action': 'f.relu'},
                                       'step4': {'action': 'layer', 'layer': 'conv2d', 'in': s1o,
                                                 'out': s4o, 'kernel_size': s4ks},
                                       'step5': {'action': 'layer', 'layer': 'conv_dropout2d'},
                                       'step6': {'action': 'f.max_pool2d', 'kernel_size': s6ks},
                                       'step7': {'action': 'f.relu'},
                                       'step8': {'action': 'view', 'dim1': -1},
                                       'step9': {'action': 'layer', 'layer': 'linear', 'in': 0, 'out': s9o},
                                       'step10': {'action': 'f.relu'},
                                       'step11': {'action': 'layer', 'layer': 'linear', 'in': s9o, 'out': 10},
                                       'step12': {'action': 'f.log_softmax', 'dim': 1}}
                                parameter_sets.append(dic)
    return parameter_sets
