import numpy as np
import pandas as pd
import collections #used for counting items of a list
from tensorflow import keras
from keras import layers
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import json

# Load Fashion MNIST dataset
(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()

input_shape = (28, 28, 1)
num_classes = 10

# Normalize input data
x_train = x_train.astype("float32") / 255
print(x_train.shape, "train samples")
x_test = x_test.astype("float32") / 255

# Reshape input data
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")

# Count number of labels in training set
nr_labels_y = collections.Counter(y_train)
print(nr_labels_y, "Number of labels")

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes)
y_labels = y_test 
y_test = keras.utils.to_categorical(y_test, num_classes)

# Flatten input data
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Define and compile model
model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

# Train model
history = model.fit(x_train, y_train, batch_size = 128, epochs = 50, validation_split = 0.1)

# Save model and weights
model.save("./data/models/model.mdl")
model.save_weights("./data/models/model.h5")

# Plot training history
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()


model = keras.models.load_model("./data/models/model.mdl")
model.load_weights("./data/models/model.h5")

# Evaluate model on test set
score = model.evaluate(x_test,y_test,verbose=2)

# Print test set accuracy
print("\n\nTest loss:", score[0])
print("Test accuracy:", score[1])

# Make predictions on test set
pred = model.predict(x_test)

# Print predicted label for first test sample
print(pred[1]) 
pred_1 = np.argmax(pred[1])
print(pred_1)

# Print true label and predicted label for first 100 test samples
for i in range(0,100):
    pred_i = np.argmax(pred[i]) 
    print (y_labels[i], pred_i)
