# IMDB Sentiment Analysis with User Input

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
vocab_size = 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

max_length = 200

# Pad sequences
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

# Build model
model = keras.Sequential([
    layers.Embedding(vocab_size, 128, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", acc)

# ================= USER INPUT =================

# Get word index (dictionary)
word_index = keras.datasets.imdb.get_word_index()

# Reverse index (word → number)
def encode_review(text):
    tokens = text.lower().split()
    encoded = []
    for word in tokens:
        if word in word_index and word_index[word] < vocab_size:
            encoded.append(word_index[word] + 3)
        else:
            encoded.append(2)  # unknown word
    return encoded

# Take input
user_review = input("\nEnter your movie review:\n")

# Encode + pad
encoded = encode_review(user_review)
padded = pad_sequences([encoded], maxlen=max_length)

# Predict
prediction = model.predict(padded)

# Output
if prediction[0][0] > 0.5:
    print("\nSentiment: Positive 😊")
else:
    print("\nSentiment: Negative 😞")
