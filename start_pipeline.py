from digit_identifier import DigitIdentifier

import additional_functions as af
from tqdm import tqdm

try_sets = False
if try_sets is True:
    forward_sets, para_sets = af.get_parameter_sets()
    print(f'Try {len(forward_sets)*len(para_sets)} different sets.')

    for para_set in tqdm(para_sets):
        for forward_set in forward_sets:
            DI = DigitIdentifier(batch_size=para_set['batch_size'], forward_dict=forward_set, info=True,
                                 loss_fn=para_set['loss_fn'], optimizer=para_set['optimizer'], lr=para_set['lr'],
                                 momentum=para_set['momentum'], weight_decay=para_set['weight_decay'])
            DI.train_model(stop_counter_max=3)
else:
    # start_- & end_index 0 - 31 -> Test des SGD optimizers
    # start_- & end_index 32 - 37 -> batch_size test
    af.plot_pandas(start_index=32, end_index=37)
    # DI = DigitIdentifier(load=True, csv_index=10)
    # DI.try_model(show=5)
