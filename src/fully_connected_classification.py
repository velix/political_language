from keras.models import Model
from keras.layers import Input, Dense
from keras import utils
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import DataLoader
import os

train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR)
test_dl = DataLoader.DataLoader(DataLoader.TEST_DATA_DIR)

input = Input(shape=(768, ))
dense1 = Dense(256, activation="relu")(input)
dense2 = Dense(1, activation="softmax")(dense1)

model = Model(inputs=input, outputs=dense2)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/fully_connected.png", show_shapes=True)

history = model.fit_generator(train_dl, epochs=10, validation_data=dev_dl)


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
plt.savefig("../models/fully_connected_accuracy.png")
plt.close()
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../models/fully_connected_loss.png")


speeches = len(os.listdir(test_dl.data_directory))
y_dev = []
model_pred = []
for speech_file in os.listdir(test_dl.data_directory):
    segments = speech_file.split("_")
    party = segments[-1][0]

    label = test_dl._party_to_label(party)
    if label is None:
        continue

    with open(os.path.join(test_dl.data_directory, speech_file), "r") as f:
        speech = f.readlines()

    sentences = test_dl._get_sentences(speech)
    doc_vectors = test_dl.bc.encode(sentences)

    predictions = model.predict(doc_vectors)
    pred_labels, pred_counts = np.unique(predictions, return_counts=True)
    predicted_label = np.argmax(pred_counts)

    y_dev.append(label)
    model_pred.append(predicted_label)

accuracy = accuracy_score(y_dev, model_pred)
print("Test accuracy: ", accuracy)
