from digit_identifier import DigitIdentifier
from catdog_classifier import CatdogClassifier
import additional_functions as af
from tqdm import tqdm
# TODO: save_all auslagern - vllt auch get optimizer
classifier = 'digit_identifier'

try_sets = False
if try_sets is True:
    forward_sets, para_sets = af.get_parameter_sets(classifier=classifier)
    print(f'Try {len(forward_sets)*len(para_sets)} different sets.')

    for para_set in tqdm(para_sets):
        for forward_set in forward_sets:
            if classifier == 'digit_identifier':
                DI = DigitIdentifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=True,
                                     loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                     momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                DI.train_model(stop_counter_max=3)
            elif classifier == 'catdog_classifier':
                CDC = CatdogClassifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=True,
                                       loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                       momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
                CDC.train_model(stop_counter_max=3)
else:
    if classifier == 'digit_identifier':
        # start_- & end_index 0 - 31 -> Test des SGD optimizers
        # start_- & end_index 32 - 38 -> batch_size test
        # af.plot_pandas(classifier=classifier, start_index=32, end_index=38)
        DI = DigitIdentifier(load=True, csv_index=10)
        DI.try_model(show=5)
    elif classifier == 'catdog_classifier':
        af.plot_pandas(classifier=classifier, start_index=0, end_index=None)
        # CDC = CatdogClassifier(load=True, csv_index=1)
        # CDC.try_model(show=5)
