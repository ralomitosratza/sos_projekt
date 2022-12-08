from digit_identifier import DigitIdentifier

# TODO: define hyperparameters and stuff
# TODO: digit_identifier loop
DI = DigitIdentifier()
for epoch in range(10):
    DI.train_model(epoch)
    DI.save_model()
# TODO: handle return values

