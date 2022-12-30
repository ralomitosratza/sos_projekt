from digit_identifier import DigitIdentifier
import additional_functions as af
from tqdm import tqdm

try_sets = False
if try_sets is True:
    forward_sets = af.get_forward_sets()
    print(f'Try {len(forward_sets)} different sets.')

    for set in tqdm(forward_sets):
        DI = DigitIdentifier(forward_dict=set, info=True)
        DI.train_model(stop_counter_max=3)
else:
    af.plot_pandas()
    DI = DigitIdentifier(load=True, csv_index=10)
    # DI.try_model(show=5)
