import numpy as np 
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout 
from tensorflow.keras.datasets import mnist 
 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print("Dataset Loaded!") 
x_train = x_train.reshape(60000, 784) 
x_test = x_test.reshape(10000, 784) 
x_train = x_train.astype('float32') / 255 
x_test = x_test.astype('float32') / 255 
y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10) 
model = Sequential() 
model.add(Dense(512, activation='relu', input_shape=(784,))) 
model.add(Dropout(0.2)) 
model.add(Dense(256, activation='relu')) 
model.add(Dropout(0.2)) 
model.add(Dense(10, activation='softmax')) 
model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(x_train, y_train, 
          epochs=5, 
          batch_size=128, 
          validation_split=0.1) 
loss, acc = model.evaluate(x_test, y_test) 
print("\nTest Accuracy:", acc) 
import matplotlib.pyplot as plt 
index = int(input("Enter test image index (0-9999): ")) 
plt.imshow(x_test[index].reshape(28,28), cmap='gray') 
plt.title("Input Image") 
plt.show() 
sample = x_test[index].reshape(1, 784) 
prediction = model.predict(sample) 
digit = np.argmax(prediction) 
print("\nPredicted Digit:", digit) 
