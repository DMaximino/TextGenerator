import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import logging


def allow_tf_memory_growth():
    # Allow memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def serialize_model(name, directory, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(directory, name + "_model.json"), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(os.path.join(directory, name + "_weights.h5"))
    logging.info("Saved model to disk")


def load_model(name, directory):
    # load json and create model
    json_file = open(os.path.join(directory, name + "_model.json"), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(os.path.join(directory, name + "_weights.h5"))
    logging.info("Loaded model from disk")

    return loaded_model


def save_history(name, directory, history):
    # Save history
    with open(os.path.join(directory, name + "_train_history.pickle"), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


def load_history(name, directory):
    return pickle.load(open(os.path.join(directory, name + "_train_history.pickle"), "rb"))


def concatenate_history(previous_history, new_history):
    pass


def plot_history(history, metrics=None):

    if metrics is None:
        metrics = ['accuracy', 'loss']

    for metric_name in metrics:
        metric = history.history[metric_name]
        epochs = range(len(metric))
        plt.plot(epochs, metric, 'b', label='Training ' + metric_name)
        plt.title('Training ' + metric_name)
        plt.figure()

    plt.show()
