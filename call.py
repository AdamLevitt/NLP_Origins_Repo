import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from data_prep import listpad_test, Forigins_test
from data_prep import tokenizer, Wordmax, dicti


def origin_name(name) -> str:
    name_input = np.array([name])
    token_input = tokenizer.texts_to_sequences(name_input)

    listpad_input = tf.keras.preprocessing.sequence.pad_sequences(
        token_input, maxlen=Wordmax, padding="pre"
    )

    origin_input = np.argmax(model.predict(listpad_input), axis=-1)
    classnum = origin_input[0]
    class_name = dicti[classnum]
    return class_name


if __name__ == "__main__":
    model = load_model(
        "/Users/lisatsakalian/Desktop/Coding_Projects/ML_Projects_Dev/NLP/model.h5"
    )

    origin_pred = origin_name("anderson")
    print(origin_pred)
