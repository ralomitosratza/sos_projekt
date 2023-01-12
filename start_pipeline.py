from digit_identifier import DigitIdentifier
from catdog_classifier import CatdogClassifier
from cifar10_classifier import Cifar10Classifier
import additional_functions as af
from tqdm import tqdm

classifier_list = ['digit_identifier', 'catdog_classifier', 'cifar10_classifier']
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
                    DI = DigitIdentifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=True,
                                         loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'],
                                         lr=para_set['lr'],
                                         momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    DI.train_model(stop_counter_max=3)
                elif classifier == 'catdog_classifier':
                    CDC = CatdogClassifier(train_data=catdog_train_data, test_data=catdog_test_data,
                                           batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                           loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'],
                                           lr=para_set['lr'],
                                           momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    catdog_train_data = CDC.train_data
                    catdog_test_data = CDC.test_data
                    CDC.train_model(stop_counter_max=3)
                elif classifier == 'cifar10_classifier':
                    C10C = Cifar10Classifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                             loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'],
                                             lr=para_set['lr'],
                                             momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    C10C.train_model(stop_counter_max=3)
elif try_sets is False:
    for classifier in classifier_list:
        if classifier == 'digit_identifier':
            # start_- & end_index 0 - 32 -> Test des SGD optimizers
            # start_- & end_index 32 - 38 -> batch_size test
            # start_- & end_index 38 - 46 -> architecture one -> konstant um 99.0 bis 99.30
            # start_- & end_index 46 - 62 -> architecture two -> konstant über 98.00 unter 99.00
            # start_- & end_index 62 - 78 -> architecture three -> konstant um 99.25 beste knapp 99.49
            # start_- & end_index 78 - 118 -> architecture four -> konstant über 98.25 unter 99.25
            af.plot_pandas(classifier=classifier, start_index=38, end_index=118, architecture=True)
            # csv_index = af.show_set(classifier=classifier, csv_index=45)
            # DI = DigitIdentifier(load=True, csv_index=csv_index)
            # DI.try_model(show=5)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='four', start_index=78, end_index=118)
            af.show_means(classifier=classifier)
        elif classifier == 'catdog_classifier':
            # start_- & end_index 0 - 8 -> architecture one
            # start_- & end_index 8 - 24 -> architecture two
            # start_- & end_index 24 - 56 -> architecture four
            # start_- & end_index 56 - 120 -> architecture six
            # start_- & end_index 248 - 256 -> architecture six
            # start_- & end_index 120 - 248 -> architecture eight
            af.plot_pandas(classifier=classifier, start_index=0, end_index=256, architecture=True)
            # csv_index = af.show_set(classifier=classifier, csv_index='highest_accuracy')
            # CDC = CatdogClassifier(load=True, csv_index=csv_index)
            # CDC.try_model(show=20)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='eight', start_index=120, end_index=248)
            af.show_means(classifier=classifier)
        elif classifier == 'cifar10_classifier':
            # start_- & end_index 0 - 8 -> architecture one
            # start_- & end_index 8 - 24 -> architecture two
            # start_- & end_index 24 - 56 -> architecture four
            # start_- & end_index 56 - 128 -> architecture six
            af.plot_pandas(classifier=classifier, start_index=0, end_index=128, architecture=True)
            # csv_index = af.show_set(classifier=classifier, csv_index='highest_accuracy')
            # C10C = Cifar10Classifier(load=True, csv_index=csv_index)
            # C10C.try_model(show=20)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='six', start_index=56, end_index=128)
            af.show_means(classifier=classifier)
