import os
import json
import numpy as np
from bert_serving.client import BertClient
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import Sequence, to_categorical


TRAINING_DATA_DIR = "../data/convote_v1.1/data_stage_one/training_set"
DEV_DATA_DIR = "../data/convote_v1.1/data_stage_one/development_set"
TEST_DATA_DIR = "../data/convote_v1.1/data_stage_one/test_set"

MAX_DOC_LENGTH = 45
MIN_DOC_LENGTH = 4

SENT_FEATURES = 768


class DataLoader:
    def __init__(self, data_directory, time_distributed=False):
        self.bc = BertClient()
        self.data_directory = data_directory
        # nlp = spacy.load('en_vectors_web_lg')
        # nlp.add_pipe(nlp.create_pipe('sentencizer'))

        self.time_distributed = time_distributed

        self.document_index = 0
        self.batch_size = 64

        self.samples = 0
        for speech_file in os.listdir(self.data_directory):
            with open(os.path.join(self.data_directory, speech_file), "r", encoding="utf8") as f:
                speech = f.readlines()

            sentences = self._get_sentences(speech)

            if len(sentences) > MIN_DOC_LENGTH:
                self.samples += 1

    def __len__(self):
        return self.samples

    def generate(self):
        speeches = os.listdir(self.data_directory)

        while True:
            vectors = []
            labels = []

            sample_counter = 0
            while sample_counter < self.batch_size:
                speech_file = speeches[self.document_index]
                if self.document_index >= len(speeches)-1:
                    self.document_index = 0
                else:
                    self.document_index += 1

                label = self._get_label(speech_file)

                doc_vectors = self._get_sentences_as_vectors(speech_file)
                # check for documents with fewer sentences than MIN_DOC_LENGTH
                if doc_vectors is None:
                    continue

                one_hot_label = to_categorical(label, 3)
                # doc_labels = np.tile(one_hot_label, [np.shape(doc_vectors)[0], 1])

                vectors.append(doc_vectors)
                labels.append(one_hot_label)
                sample_counter += 1

            # if self.data_directory == TEST_DATA_DIR:
            #    print("batch")

            yield (np.asarray(vectors), np.asarray(labels))

    def _get_label(self, speech_file):
        segments = speech_file.split("_")
        party = segments[-1][0]

        return self._party_to_label(party)

    def _get_sentences_as_vectors(self, speech_file):
        with open(os.path.join(self.data_directory, speech_file), "r", encoding="utf8") as f:
            speech = f.readlines()

        sentences = self._get_sentences(speech)
        try:
            doc_vectors = self.bc.encode(sentences)
            doc_vectors = self._pad_document(doc_vectors)
            return doc_vectors

        except (UnicodeDecodeError, json.decoder.JSONDecodeError) as e:
            return None


    def _pad_document(self, doc_vectors):
        sentences, words = np.shape(doc_vectors)
        if sentences < MIN_DOC_LENGTH:
            return None
        if sentences > MAX_DOC_LENGTH:
            return doc_vectors[:MAX_DOC_LENGTH, :]

        # Padding the doc_vectors to have MAX_DOC_LENGTH
        # number of rows. Padding with zeros
        pad_length = MAX_DOC_LENGTH - sentences
        return np.pad(doc_vectors, [(0, pad_length), (0,0)], 'constant')


    def _party_to_label(self, party):
        if party == "D":
            return 1
        elif party == "R":
            return 0
        else:
            return -1

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
            sentence = sentence.replace(sentence[open_par], '')
            
            close_par = sentence.find(")")
            sentence = sentence.replace(sentence[close_par], '')

            ref_number_start = sentence.find("xz")
            ref_number = sentence[ref_number_start:ref_number_start+9]

            sentence = sentence.replace(ref_number, '')

        return sentence



