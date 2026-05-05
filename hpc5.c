import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
 
print("GPU Available:", tf.config.list_physical_devices('GPU')) 
 
epochs = int(input("Enter number of epochs: ")) 
batch_size = int(input("Enter batch size: ")) 
 
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data() 
 
x_train = x_train / 255.0 
x_test = x_test / 255.0 
 
model = keras.Sequential([ 
    keras.layers.Flatten(input_shape=(28, 28)), 
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dense(10, activation='softmax') 
]) 
 
model.compile( 
    optimizer='adam', 
    loss='sparse_categorical_crossentropy', 
    metrics=['accuracy'] 
) 
 
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size) 
 
loss, acc = model.evaluate(x_test, y_test) 
print("Accuracy:", acc)
