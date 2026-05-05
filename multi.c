mport tensorflow as tf 
from tensorflow import keras 
import numpy as np 
 
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data() 
print("Dataset Loaded Successfully!") 
x_train = x_train.astype('float32') / 255 
x_test = x_test.astype('float32') / 255 
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 
model = keras.Sequential([ 
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)), 
    keras.layers.MaxPooling2D((2,2)), 
    keras.layers.Dropout(0.25), 
    keras.layers.Conv2D(64, (3,3), activation='relu'), 
    keras.layers.MaxPooling2D((2,2)), 
    keras.layers.Dropout(0.25), 
    keras.layers.Conv2D(128, (3,3), activation='relu'), 
    keras.layers.Flatten(), 
    keras.layers.Dense(128, activation='relu'), 
    keras.layers.Dropout(0.25), 
    keras.layers.Dense(10, activation='softmax') 
]) 
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']) 
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test)) 
loss, acc = model.evaluate(x_test, y_test) 
print("\nTest Accuracy:", acc) 
class_names = [ 
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot" 
] 
print("\nChoose option:") 
print("1. Use test sample") 
print("2. Enter custom image pixels") 
choice = int(input("Enter choice (1 or 2): ")) 
if choice == 1: 
    index = int(input("Enter test image index (0-9999): ")) 
    sample = x_test[index].reshape(1, 28, 28, 1) 
else: 
    print("Enter 784 pixel values (0–255):") 
    pixels = [] 
    for i in range(784): 
        val = float(input(f"Pixel {i+1}: ")) 
        pixels.append(val) 
    sample = np.array(pixels).reshape(1, 28, 28, 1) / 255 
prediction = model.predict(sample) 
