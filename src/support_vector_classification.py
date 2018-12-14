from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import DataLoader


dl = DataLoader.DataLoader()

X = []
y = []

for X, y in dl.generate_X(DataLoader.TRAINING_DATA_DIR):
    X.extend(X)
    y.extend(y)

svc = SVC(gamma="scale").fit(X, y)

predictions = svc.predict(X)

accuracy = accuracy_score(y, predictions)
print("SVC Training accuracy: ", accuracy)

# predictions = []
# for speech in range(len(y)):
#     predictions.append(np.random.choice(["D", "R"], 1))

# accuracy = accuracy_score(y, predictions)
# print("Random Training accuracy: ", accuracy)

X_dev = []
y_dev = []

for X_dev, y_dev in dl.generate_X(DataLoader.DEV_DATA_DIR):
    X_dev.extend(X_dev)
    y_dev.extend(y_dev)

predictions = svc.predict(X_dev)

accuracy = accuracy_score(y_dev, predictions)
print("SVC Dev accuracy: ", accuracy)