from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import utils
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import DataLoader
import os
import json

train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR)
test_dl = DataLoader.DataLoader(DataLoader.TEST_DATA_DIR)

print("Train samples: ", train_dl.samples)
print("Dev samples: ", dev_dl.samples)
print("Test samples: ", test_dl.samples)

input = Input(shape=(DataLoader.MAX_DOC_LENGTH, DataLoader.SENT_FEATURES))
dense0 = Dense(512, activation="relu")(input)
dropout = Dropout(0.5)(dense0)
dense1 = Dense(256, activation="relu")(dropout)
dropout = Dropout(0.5)(dense1)
dense2 = Dense(1, activation="softmax")(dropout)

model = Model(inputs=input, outputs=dense2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/fully_connected.png", show_shapes=True)

callback_chkpt = ModelCheckpoint(
                       "../models/fully_connected.hdf5",
                       monitor='val_categorical_accuracy', verbose=1,
                       save_best_only=True, mode='max')
callback_stopping = EarlyStopping(monitor="val_categorical_accuracy",
                                  mode="max", patience=2)

history = model.fit_generator(train_dl.generate(), epochs=10,
                              validation_data=dev_dl.generate(),
                              validation_steps=int(dev_dl.samples/dev_dl.batch_size),
                              steps_per_epoch=int(train_dl.samples/train_dl.batch_size),
                              callbacks=[callback_chkpt, callback_stopping])

with open("../results/fully_connected_history.json", "w") as f:
    json.dump(history.history, f)

score = model.evaluate_generator(test_dl.generate(),
                                 steps=int(test_dl.samples/test_dl.batch_size))


print("Test loss {}, test cat. accuracy: {} ".format(score[0], score[1]))


# summarize history for accuracy
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../results/fully_connected_accuracy.png")
plt.close()
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../results/fully_connected_loss.png")


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
