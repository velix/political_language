from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense
import numpy as np
import DataLoader

dl = DataLoader.DataLoader()

input = Input(shape=(, 768))
gru = GRU(units=100)(input)
dense = Dense(1, activation="softmax")(gru)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

for epoch in range(20):
    for (X, y) in dl.generate_vectors(DataLoader.TRAINING_DATA_DIR):
        (X_dev, y_dev) = next(dl.generate_vectors(DataLoader.DEV_DATA_DIR))
        # X = np.expand_dims(X, axis=0)
        # X_dev = np.expand_dims(X_dev, axis=0)
        print("X: ", np.shape(X))
        print("y:", np.shape(y))
        print("X_dev: ", np.shape(X_dev))
        print("y_dev: ",np.shape(y_dev))
        hist = model.fit(X, y, batch_size=np.shape(X)[0], epochs=1)

score = model.evaluate_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

predictions = model.predict_generator(dl.generate_vectors(DataLoader.DEV_DATA_DIR))

print("Final model score: ", score)
