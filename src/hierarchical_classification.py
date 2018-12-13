from keras.models import Sequential
from keras.layers import GRU, Bidirectional, Dense

import DataLoader

dl = DataLoader.DataLoader()

model = Sequential()

model.add(Bidirectional(GRU(units=50), input_shape=(None, 768)))
model.add(Dense(1, activation="softmax"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

for epoch in range(20):
    for (X, y) in dl.generate_vectors(DataLoader.TRAINING_DATA_DIR):
        (X_dev, y_dev) = next(dl.generate_vectors(DataLoader.DEV_DATA_DIR))
        hist = model.fit(X, y, batch_size=len(X), epochs=1)

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)