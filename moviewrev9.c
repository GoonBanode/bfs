#Classify movie reviews into "positive" reviews and "negative" reviews, just based on the text content of the reviews. Use IMDB dataset.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load IMDB dataset (top 10,000 words)
vocab_size = 10000

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

# Pad sequences to same length
max_length = 200

x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_length)

# Build model
model = keras.Sequential([
    layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')   # Binary output
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# Evaluate
loss, acc = model.evaluate(x_test, y_test)
print("\nTest Accuracy:", acc)

# Predict example
sample = x_test[0:1]
prediction = model.predict(sample)

print("Prediction:", "Positive" if prediction[0][0] > 0.5 else "Negative")
print("Actual:", "Positive" if y_test[0] == 1 else "Negative")
