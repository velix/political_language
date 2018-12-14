from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense, TimeDistributed
from keras import utils 
import DataLoader

train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR, True)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR, True)
test_dl = DataLoader.DataLoader(DataLoader.TEST_DATA_DIR, True)

input = Input(shape=(768, ))
# gru = GRU(units=100, return_sequences=True, stateful=True)(input)
encoded = TimeDistributed(Bidirectional(GRU(units=50, return_sequences=True, stateful=True)))(input)

dense = Dense(1, activation="softmax")(encoded)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["binary_accuracy", "categorical_accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/bidirectional_gru.png", show_shapes=True)

history = model.fit_generator(train_dl, epochs=4, validation_data=dev_dl)

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)
