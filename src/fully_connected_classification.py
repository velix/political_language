from keras.models import Model
from keras.layers import Input, Dense
from keras import utils
import matplotlib.pyplot as plt
import DataLoader

train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR)

input = Input(shape=(768, ))
dense1 = Dense(256, activation="relu")(input)
dense2 = Dense(1, activation="softmax")(dense1)

model = Model(inputs=input, outputs=dense2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/fully_connected.png", show_shapes=True)

history = model.fit_generator(train_dl, epochs=1, validation_data=dev_dl)


# for epoch in range(1):
#     for (X, y) in dl.generate_vectors(DataLoader.TRAINING_DATA_DIR):
#         (X_dev, y_dev) = next(dl.generate_vectors(DataLoader.DEV_DATA_DIR))
#         hist = model.fit(X, y, batch_size=len(X), epochs=1)

score = model.evaluate_generator(dev_dl)

predictions = model.predict_generator(dev_dl)

print("Final model score: ", score)


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
