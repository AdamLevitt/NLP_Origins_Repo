import glob2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import encode_func

# Combine and define data sets
together = glob2.glob("data/names/*.txt")
data = []

for filename in together:
    origin = filename.split("/")[-1].split(".txt")[0]
    names = open(filename).readlines()
    for name in names:
        data.append((name.strip(), origin))

names, origins = zip(*data)

names_train, names_test, origins_train, origins_test = train_test_split(
    names, origins, test_size=0.25, shuffle=True, random_state=123
)

# Filter out 'to The First Page' from data sets
length = len(names_train)
names_traina = []
origins_traina = []

for index in range(length):

    if names_train[index] != "To The First Page":
        names_traina.append(names_train[index])
        origins_traina.append(origins_train[index])

tlength = len(names_test)
names_testa = []
origins_testa = []

for indext in range(tlength):

    if names_test[indext] != "To The First Page":
        names_testa.append(names_test[indext])
        origins_testa.append(origins_test[indext])


# Convert to numpy arrays
Nnames_train = np.array(names_traina)
Norigins_train = np.array(origins_traina)

Nnames_test = np.array(names_testa)
Norigins_test = np.array(origins_testa)

# Find the number of unique label values (origins)
a = np.unique(Norigins_test)
b = np.unique(Norigins_train)

# integer Encode Labels (Origins)
Torigins_test = encode_func.int_encode(Norigins_test)
Torigins_train = encode_func.int_encode(Norigins_train)

# Create a dictionary from integer values back to origin 'Text' label
dicti = {}

for count in range(len(Torigins_train)):
    dicti[Torigins_train[count]] = Norigins_train[count]


# One Hot-Encode
Forigins_test = to_categorical(Torigins_test)
Forigins_train = to_categorical(Torigins_train)


# Fit tokenizer to names at a character level
tokenizer = keras.preprocessing.text.Tokenizer(char_level=True, lower=True)
tokenizer.fit_on_texts(Nnames_train)
tokenizer.fit_on_texts(Nnames_test)

# Number of characters
char = tokenizer.word_index
char_length = len(char)

# Convert names to token values
token = tokenizer.texts_to_sequences(Nnames_train)
token_test = tokenizer.texts_to_sequences(Nnames_test)

# Find the max word length
WordmaxTR = max([len(temp) for temp in token])
WordmaxTE = max([len(temp) for temp in token_test])

Wordmax = max(WordmaxTE, WordmaxTR)

# Pad tokenized words
listpad = tf.keras.preprocessing.sequence.pad_sequences(
    token, maxlen=Wordmax, padding="pre"
)
listpad_test = tf.keras.preprocessing.sequence.pad_sequences(
    token_test, maxlen=Wordmax, padding="pre"
)