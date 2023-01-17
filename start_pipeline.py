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
        # Die zu testenden Architekturen bzw. weiteren Parameter werden ausgelesen
        forward_sets, para_sets = af.get_parameter_sets(classifier=classifier)
        print(f'Try {len(forward_sets)*len(para_sets)} different sets for {classifier}.')
        for forward_set in tqdm(forward_sets):
            for para_set in para_sets:
                if classifier == 'digit_identifier':
                    # DNN wird für jedes Set neu erstellt, trainiert und getestet
                    DI = DigitIdentifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=True,
                                         loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'],
                                         lr=para_set['lr'],
                                         momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    DI.train_model(stop_counter_max=3)
                elif classifier == 'catdog_classifier':
                    # DNN wird für jedes Set neu erstellt, trainiert und getestet
                    CDC = CatdogClassifier(train_data=catdog_train_data, test_data=catdog_test_data,
                                           batch_size=para_set['batch_size'], forward_dict=forward_set, info=False,
                                           loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'],
                                           lr=para_set['lr'],
                                           momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                    catdog_train_data = CDC.train_data
                    catdog_test_data = CDC.test_data
                    CDC.train_model(stop_counter_max=3)
                elif classifier == 'cifar10_classifier':
                    # DNN wird für jedes Set neu erstellt, trainiert und getestet
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
            # start_- & end_index 38 - 46 -> architecture one
            # start_- & end_index 46 - 62 -> architecture two
            # start_- & end_index 62 - 78 -> architecture three
            # start_- & end_index 78 - 118 -> architecture four

            # plottet die Ergebnisse der entsprechenden DNNs vom start_index bis zum end_index.
            # af.plot_pandas(classifier=classifier, start_index=38, end_index=118)

            # printet die Architektur des entsprechenden DNNs an csv_index-Stelle. Gibt diesen zurück zur Weiterverarbeitung.
            # alternativ kann auch 'highest_accuracy' eigegeben werden. Dann wird das Netz mit der höchsten Genauigkeit verwendet.
            csv_index = af.show_set(classifier=classifier, csv_index='highest_accuracy')

            # läd das entsprechende, trainierte Netz an csv_index-Stelle.
            DI = DigitIdentifier(load=True, csv_index=csv_index)

            # lässt einen visuell das Netz testen.
            DI.try_model(show=5)

            # speichert Durchschnitte zu einer csv-Datei (ist bereits geschehen)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='four', start_index=78, end_index=118)

            # zeigt die Durchschnitte in einer Grafik
            # af.show_means(classifier=classifier)

        elif classifier == 'catdog_classifier':
            # start_- & end_index 0 - 8 -> architecture one
            # start_- & end_index 8 - 24 -> architecture two
            # start_- & end_index 24 - 56 -> architecture four
            # start_- & end_index 56 - 120 -> architecture five
            # start_- & end_index 248 - 256 -> architecture five
            # start_- & end_index 120 - 248 -> architecture six
            # af.plot_pandas(classifier=classifier, start_index=0, end_index=256)
            csv_index = af.show_set(classifier=classifier, csv_index='highest_accuracy')
            CDC = CatdogClassifier(load=True, csv_index=csv_index)
            CDC.try_model(show=20)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='eight', start_index=120, end_index=248)
            # af.show_means(classifier=classifier)
        elif classifier == 'cifar10_classifier':
            # start_- & end_index 0 - 8 -> architecture one
            # start_- & end_index 8 - 24 -> architecture two
            # start_- & end_index 24 - 56 -> architecture four
            # start_- & end_index 56 - 128 -> architecture five
            # af.plot_pandas(classifier=classifier, start_index=0, end_index=128)
            csv_index = af.show_set(classifier=classifier, csv_index='highest_accuracy')
            C10C = Cifar10Classifier(load=True, csv_index=csv_index)
            C10C.try_model(show=20)
            # af.save_means_of_csv_to_csv(classifier=classifier, architecture='five', start_index=56, end_index=128)
            # af.show_means(classifier=classifier)
