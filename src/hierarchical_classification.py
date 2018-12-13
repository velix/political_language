from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense

import DataLoader

dl = DataLoader.DataLoader()

input = Input(batch_shape=(None, 384))
gru = GRU(units=100)(input)
dense = Dense(1, activation="softmax")(gru)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

for epoch in range(20):
    for (X, y) in dl.generate_vectors(DataLoader.TRAINING_DATA_DIR):
        (X_dev, y_dev) = next(dl.generate_vectors(DataLoader.DEV_DATA_DIR))
        hist = model.fit(X, y, batch_size=len(X), epochs=1)

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)
