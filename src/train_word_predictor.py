from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.utils as ku
import tensorflow as tf
from constants import *
import utilities
import nlp_helpers
import models
import json

utilities.allow_tf_memory_growth()

with open(CONFIG_FILE_PATH) as json_file:
    config = json.load(json_file)

# Load corpus into input sequences
corpus = nlp_helpers.load_corpus(config["corpus_path"])
tokenizer = Tokenizer(num_words=config["num_words"], oov_token=None)
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
input_sequences = nlp_helpers.create_input_sequences(corpus, tokenizer, random_sampling=220000)

# create predictors and label
predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
label = ku.to_categorical(label, num_classes=total_words)

if config["start_from_pretrained_model"]:
    model = utilities.load_model(config["pretrained_model_name"], config["pretrained_model_dir"])
    previous_history = utilities.load_history(config["pretrained_model_name"], config["pretrained_model_dir"])
else:
    model = models.word_predictor_model(config["batch_size"], total_words, len(input_sequences[0]))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(predictors, label, batch_size=config["batch_size"], epochs=config["epochs"], verbose=1)

utilities.plot_history(history)

utilities.serialize_model(config["save_name"], config["save_dir"], model)
utilities.save_history(config["save_name"], config["save_dir"], history)



