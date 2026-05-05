import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000) 
def vectorize(sequences, dimension=10000): 
    results = np.zeros((len(sequences), dimension)) 
    for i, seq in enumerate(sequences): 
        results[i, seq] = 1 
    return results 
X_train = vectorize(X_train) 
X_test = vectorize(X_test) 
y_train = np.array(y_train).astype("float32") 
y_test = np.array(y_test).astype("float32") 
model = Sequential() 
model.add(Dense(64, activation='relu', input_shape=(10000,))) 
model.add(Dropout(0.3)) 
model.add(Dense(32, activation='relu')) 
model.add(Dropout(0.3)) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy']) 
model.fit(X_train, y_train, 
          epochs=5, 
          batch_size=512, 
          validation_split=0.2) 
loss, acc = model.evaluate(X_test, y_test) 
print("\nTest Accuracy:", acc) 
word_index = imdb.get_word_index() 
def encode_review(text): 
    words = text.lower().split() 
    encoded = [] 
    for word in words: 
        index = word_index.get(word, 2)  # 2 = unknown word 
        encoded.append(index + 3)        # offset 
    return encoded 
review = input("\nEnter your movie review: ") 
encoded_review = encode_review(review) 
vector = vectorize([encoded_review]) 
prediction = model.predict(vector) 
if prediction[0][0] > 0.5: 
    print("\nResult: Positive Review       ") 
else: 
    print("\nResult: Negative Review     ") 
