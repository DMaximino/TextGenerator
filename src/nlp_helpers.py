import numpy as np
from random import sample
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_corpus(path):
    raw_corpus = open(path).read()
    corpus = raw_corpus.lower().split("\n")
    corpus = np.array(corpus)
    return corpus


def create_input_sequences(corpus, tokenizer, padding='pre', random_sampling=None):
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]

        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    if random_sampling is not None:
        input_sequences = sample(list(input_sequences), random_sampling)

    max_sequence_len = max([len(x) for x in input_sequences])
    return np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding=padding))