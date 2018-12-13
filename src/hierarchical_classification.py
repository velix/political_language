from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense

import DataLoader

dl = DataLoader.DataLoader()

model = Sequential()

model.add(Bidirectional(GRU(units=50)), input_shape=(1, 384))
model.add(Dense(1, activation="softmax"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

hist = model.fit_generator(dl.generate_vectors(DataLoader.TRAINING_DATA_DIR),
                           validation_data=dl.generate_vectors(DataLoader.DEV_DATA_DIR),
                           epochs=20, steps_per_epoch=len(dl.generate_vectors(DataLoader.TRAINING_DATA_DIR)))

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)