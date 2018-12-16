from keras.models import Model
from keras.layers import Input, GRU, Bidirectional, Dense, TimeDistributed
from keras import utils
import DataLoader
import matplotlib.pyplot as plt
import numpy as np

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
# encoded = GRU(units=100)(input)
bidirectional_gru = Bidirectional(GRU(units=50))(input)
# encoded = TimeDistributed(gru)(input)

dense = Dense(3, activation="softmax")(bidirectional_gru)

model = Model(inputs=input, outputs=dense)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

utils.print_summary(model)
utils.plot_model(model, to_file="../models/bidirectional_gru.png", show_shapes=True)

history = model.fit_generator(train_dl.generate(), epochs=1,
                              validation_data=dev_dl.generate(),
                              validation_steps=int(dev_dl.samples/dev_dl.batch_size),
                              steps_per_epoch=int(train_dl.samples/train_dl.batch_size))

score = model.evaluate_generator(test_dl.generate(),
                                 steps=int(test_dl.samples/test_dl.batch_size))

predictions = model.predict_generator(test_dl.generate(),
                                      steps=int(test_dl.samples/test_dl.batch_size))

print("Final model score: ", score)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../models/hierarchical_bidirectional_accuracy.png")
plt.close()
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("../models/hierarchical_bidirectional_loss.png")
