# Build Model

import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM
from keras.optimizers import Adam
from data_prep import listpad, listpad_test, Forigins_train, Forigins_test, char_length, Wordmax

# Model makeup
def model_form():

    model = keras.models.Sequential(
        [
            keras.layers.Embedding(char_length + 1, 100, input_length=Wordmax),
            keras.layers.Bidirectional(LSTM(75)),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(75, activation="relu"),
            keras.layers.Dense(18, activation="softmax"),
        ]
    )

    # Define Callbacks
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3
    )

    # Compile & Show Summary
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    return model, earlyStopping_callback

#Model summary view
def model_print(funct):
    print("\n")
    mod, *stop = funct()
    print(mod.summary())
    return mod, stop

#Model Compile
def compile_mod(function):
    print("\n")
    mod1, *stop1 = function()
    history = mod1.fit(
        listpad,
        Forigins_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[stop1],
        verbose=2,
        batch_size=32,
    )
    return history

#Print & Compile NLP Model
printmod = model_print(model_form)
history = compile_mod(model_form)