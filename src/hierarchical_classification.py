from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense, Dropout, TimeDistributed
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import utils
import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json


train_dl = DataLoader.DataLoader(DataLoader.TRAINING_DATA_DIR, True)
dev_dl = DataLoader.DataLoader(DataLoader.DEV_DATA_DIR, True)
test_dl = DataLoader.DataLoader(DataLoader.TEST_DATA_DIR, True)

print("Train samples: ", train_dl.samples)
print("Dev samples: ", dev_dl.samples)
print("Test samples: ", test_dl.samples)

'''
RNN input must take the form (batch_size, timesteps, features(hidden size/units))
in this case, each timestep is a sentence

we want an output of the form (batch_size, units), to represent a set of
sentences (the speech) as a single vector. We need the output of
the last node in the rnn
'''

input = Input(shape=(DataLoader.MAX_DOC_LENGTH, DataLoader.SENT_FEATURES))
bidirectional_gru = Bidirectional(GRU(units=190))(input)
dropout = Dropout(0.5)(bidirectional_gru)
dense = Dense(3, activation="softmax")(dropout)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/hierarchical_bidirectional.png", show_shapes=True)

callback_chkpt = ModelCheckpoint(
                       "../models/hierarchical_bidirectional.hdf5",
                       monitor='val_categorical_accuracy', verbose=1,
                       save_best_only=True, mode='max')
callback_stopping = EarlyStopping(monitor="val_categorical_accuracy",
                                  mode="max", patience=1)

if False:
        history = model.fit_generator(train_dl.generate(), epochs=10,
                                      validation_data=dev_dl.generate(),
                                      validation_steps=int(dev_dl.samples/dev_dl.batch_size),
                                      steps_per_epoch=int(train_dl.samples/train_dl.batch_size),
                                      callbacks=[callback_chkpt, callback_stopping])

        with open("../results/bidirectional_hierarchical_history.json", "w") as f:
                json.dump(history.history, f)

history = json.load(open("../results/bidirectional_hierarchical_history.json", "r"))
model = load_model("../models/hierarchical_bidirectional.hdf5")
score = model.evaluate_generator(test_dl.generate(),
                                 steps=int(test_dl.samples/test_dl.batch_size))

# predictions = model.predict_generator(test_dl.generate(),
#                                      steps=int(test_dl.samples/test_dl.batch_size))

print("Test loss {}, test cat. accuracy: {} ".format(score[0], score[1]))

print("Plotting...")
# summarize history for accuracy
plt.plot(history['categorical_accuracy'])
plt.plot(history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../results/hierarchical_bidirectional_accuracy.png")
plt.close()
plt.clf()

# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../results/hierarchical_bidirectional_loss.png")

print("Done")
