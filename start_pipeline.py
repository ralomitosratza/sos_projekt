from digit_identifier import DigitIdentifier
import additional_functions as af
from tqdm import tqdm

epochs = 10
forward_sets = af.get_forward_set()
print(f'Try {len(forward_sets)} different sets.')


for set in tqdm(forward_sets):
    DI = DigitIdentifier(forward_dict=set)
    DI.train_model(epochs)
    # DI.show_results(show=5)


