import os
# import spacy
import numpy as np
from bert_serving.client import BertClient
from keras.preprocessing.text import text_to_word_sequence


TRAINING_DATA_DIR = "../data/convote_v1.1/data_stage_one/training_set"
DEV_DATA_DIR = "../data/convote_v1.1/data_stage_one/development_set"
TEST_DATA_DIR = "../data/convote_v1.1/data_stage_one/test_set"


class DataLoader:
    def __init__(self):
        self.bc = BertClient()
        # nlp = spacy.load('en_vectors_web_lg')
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))

        self.vectors = []
        self.labels = []

    def generate_vectors(self, data_directory):
        for idx, speech_file in enumerate(os.listdir(data_directory)):
            # ###_@@@@@@_%%%%$$$_PMV
            segments = speech_file.split("_")
            party = segments[-1][0]

            if party == "X":
                continue
            label = self._party_to_label(party)

            with open(os.path.join(data_directory, speech_file), "r") as f:
                speech = f.readlines()

            sentences = self._get_sentences(speech)
            doc_vectors = self.bc.encode(sentences)
            doc_labels = np.ones((np.shape(doc_vectors)[0],))*label

            yield(doc_vectors, doc_labels)

            # labels = label_binarize(labels, np.unique(np.array(labels, dtype=str)))

    def _party_to_label(self, party):
        if party == "D":
            return 1
        elif party == "R":
            return 0


    def _get_sentences(self, speech_lines):
        speech_doc = "".join(speech_lines)

        sentences = []
        for sentence in speech_doc.split("."):
            sentence = sentence.strip()
            sentence = self._clean_up_names(sentence)
            sentence = " ".join(text_to_word_sequence(sentence))

            if len(sentence) > 0:
                sentences.append(sentence)

        return sentences

    def _clean_up_names(self, sentence):
        open_par = sentence.find("(")
        if open_par != -1:
            close_par = sentence.find(")")

            ref_number_start = sentence.find("xz")
            ref_number = sentence[ref_number_start:ref_number_start+9]

            sentence = sentence.replace(ref_number, '')
            sentence = sentence.replace(sentence[open_par], '')
            sentence = sentence.replace(sentence[close_par], '')

        return sentence



