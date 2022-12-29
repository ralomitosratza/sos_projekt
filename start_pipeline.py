from digit_identifier import DigitIdentifier
import additional_functions as af
from tqdm import tqdm

try_sets = False
if try_sets is True:
    epochs = 10
    forward_sets = af.get_forward_set()
    print(f'Try {len(forward_sets)} different sets.')

    for set in tqdm(forward_sets):
        DI = DigitIdentifier(forward_dict=set, epochs=epochs)
        DI.train_model(epochs)
        # TODO: Erst beenden, wenn ausgelernt.
else:
    af.plot_pandas()
    DI = DigitIdentifier(load=True, csv_index=10)
    # DI.try_model(show=5)
