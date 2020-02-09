from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional


def word_predictor_model(batch_size, total_words, max_sequence_len):
    # Define model
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1,
                        batch_input_shape=(batch_size, max_sequence_len - 1)))
    model.add(Bidirectional(LSTM(150, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dense(total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(total_words, activation='softmax'))
    print(model.summary())
    return model