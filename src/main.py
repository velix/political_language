from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
import numpy as np
import os
import spacy

TRAINING_DATA_DIR = "data/convote_v1.1/data_stage_one/training_set"

nlp = spacy.load('en_vectors_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'))

vectors = []
labels = []

speeches = len(os.listdir(TRAINING_DATA_DIR))
for idx, speech_file in enumerate(os.listdir(TRAINING_DATA_DIR)):
    print("processing {}/{}".format(idx+1, speeches))

    # ###_@@@@@@_%%%%$$$_PMV
    segments = speech_file.split("_")
    party = segments[-1][0]

    with open(os.path.join(TRAINING_DATA_DIR, speech_file), "r") as f:
        speech = f.readlines()

    speech_doc = nlp("".join(speech))
    sentences = speech_doc.sents
    vector = np.mean([sent.vector for sent in sentences], axis=0)

    vectors.append(vector)
    labels.append(party)

# labels = label_binarize(labels, np.unique(np.array(labels, dtype=str)))
vectors = np.array(vectors)


svc = SVC(gamma="scale").fit(vectors, labels)

predictions = svc.predict(vectors)

accuracy = accuracy_score(labels, predictions)
print("SVC Training accuracy: ", accuracy)

predictions = []
for speech in range(len(labels)):
    predictions.append(np.random.choice(["D", "R"], 1))

accuracy = accuracy_score(labels, predictions)
print("Random Training accuracy: ", accuracy)