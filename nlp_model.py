# Build model & save file to directory

import numpy as np
import tensorflow as tf
import os
from pathlib import Path
from tensorflow import keras
from keras.layers import LSTM
from keras.optimizers import Adam
from data_prep import (
    listpad,
    listpad_test,
    Forigins_train,
    Forigins_test,
    char_length,
    Wordmax,
)

# Model forming
def model_form():

    global model

    model = keras.models.Sequential(
        [
            keras.layers.Embedding(char_length + 1, 100, input_length=Wordmax),
            keras.layers.Bidirectional(LSTM(75)),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(75, activation="relu"),
            keras.layers.Dense(18, activation="softmax"),
        ]
    )

    # Define callbacks
    earlyStopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=3
    )

    # Compile
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    return model, earlyStopping_callback


# Model summary view
def model_print(funct):
    print("\n")
    mod, *stop = funct()
    print(mod.summary())
    return mod, stop


# Define paths
local = os.getcwd() + "/"
accuracy_store = local + "accuracy.md"

# Creates accuracy file is not already existing
with open(accuracy_store, "a"):
    pass

accuracy = 0

# Pull past accuracy that was stored
try:
    with open(accuracy_store, "r") as file:
        for line in file:
            accuracy = float(line)

except Exception:
    print("error")

# Call and fit model
mod1, *stop1 = model_form()
h = mod1.fit(
    listpad,
    Forigins_train,
    epochs=50,
    validation_split=0.2,
    callbacks=[stop1],
    verbose=2,
    batch_size=32,
)

# Evaluate test data
loss, acc = mod1.evaluate(listpad_test, Forigins_test, verbose=0)

# Compare test accuracy versus previous figure - update and save model if improved
print("\n")
print(f"acc: {acc}")
print(f"accuracy: {accuracy}")
if acc > accuracy:
    print("Model Improved")
    with open(accuracy_store, "w") as file:
        file.write("%f" % acc)

    model.save("model.h5")
    print("Saved model to disk")

else:
    print("Model is not improved")