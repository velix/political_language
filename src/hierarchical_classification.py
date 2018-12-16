from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense, TimeDistributed
from keras import utils 
import DataLoader

train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR, True)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR, True)
test_dl = DataLoader.DataLoader(DataLoader.TEST_DATA_DIR, True)


'''
RNN input must take the form (batch_size, timesteps, features(hidden size/units))
in this case, each timestep is a sentence

we want an output of the form (batch_size, units), to represent a set of
sentences (the speech) as a single vector. We need the output of
the last node in the rnn
'''

input = Input(shape=(DataLoader.MAX_DOC_LENGTH, DataLoader.SENT_FEATURES))
encoded = GRU(units=100)(input)
# bidirectional_gru = Bidirectional(GRU(units=50, return_sequences=True))
# encoded = TimeDistributed(gru)(input)

dense = Dense(3, activation="softmax")(encoded)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/bidirectional_gru.png", show_shapes=True)

history = model.fit_generator(train_dl, epochs=4, validation_data=dev_dl)

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)
