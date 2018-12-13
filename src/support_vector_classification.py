from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from DataLoader import DataLoader


loader = DataLoader()

svc = SVC(gamma="scale").fit(vectors, labels)

predictions = svc.predict(vectors)

accuracy = accuracy_score(labels, predictions)
print("SVC Training accuracy: ", accuracy)

predictions = []
for speech in range(len(labels)):
    predictions.append(np.random.choice(["D", "R"], 1))

accuracy = accuracy_score(labels, predictions)
print("Random Training accuracy: ", accuracy)
