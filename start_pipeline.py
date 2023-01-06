from digit_identifier import DigitIdentifier
from catdog_classifier import CatdogClassifier
from cifar10_classifier import Cifar10Classifier
import additional_functions as af
from tqdm import tqdm

classifier_list = ['digit_identifier']
catdog_train_data = None
catdog_test_data = None

try_sets = False
if try_sets is True:
    for classifier in classifier_list:
        forward_sets, para_sets = af.get_parameter_sets(classifier=classifier)
        print(f'Try {len(forward_sets)*len(para_sets)} different sets for {classifier}.')
        for forward_set in tqdm(forward_sets):
            for para_set in para_sets:
                if classifier == 'digit_identifier':
                    DI = DigitIdentifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                         loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                         momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    DI.train_model(stop_counter_max=3)
                elif classifier == 'catdog_classifier':
                    CDC = CatdogClassifier(train_data=catdog_train_data, test_data=catdog_test_data,
                                           batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                           loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                           momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    catdog_train_data = CDC.train_data
                    catdog_test_data = CDC.test_data
                    CDC.train_model(stop_counter_max=3)
                elif classifier == 'cifar10_classifier':
                    C10C = Cifar10Classifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                             loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                             momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    C10C.train_model(stop_counter_max=3)
elif try_sets is False:
    for classifier in classifier_list:
        if classifier == 'digit_identifier':
            # start_- & end_index 0 - 32 -> Test des SGD optimizers
            # start_- & end_index 32 - 38 -> batch_size test
            # start_- & end_index 38 - 46 -> architecture one
            # start_- & end_index 46 - 62 -> architecture two
            # start_- & end_index 62 - 78 -> architecture three
            # start_- & end_index 78 - 110 -> architecture four
            af.plot_pandas(classifier=classifier, start_index=38, end_index=110)
            af.show_set(classifier=classifier, csv_index=0)
            # DI = DigitIdentifier(load=True, csv_index=10)
            # DI.try_model(show=5)
        elif classifier == 'catdog_classifier':
            # start_- & end_index 0 - 8 -> architecture one
            # start_- & end_index 8 - 24 -> architecture two
            # start_- & end_index 24 - 40 -> architecture four
            af.plot_pandas(classifier=classifier, start_index=0, end_index=40)
            af.show_set(classifier=classifier, csv_index=0)
            # CDC = CatdogClassifier(load=True, csv_index=1)
            # CDC.try_model(show=20)
        elif classifier == 'cifar10_classifier':
            af.plot_pandas(classifier=classifier, start_index=0, end_index=None)
            af.show_set(classifier=classifier, csv_index=0)
            # C10C = Cifar10Classifier(load=True, csv_index=1)
            # C10C.try_model(show=20)
