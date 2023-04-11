import numpy as np
import pandas as pd
import collections #used for counting items of a list
from keras.datasets import mnist, fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
from PIL import Image
import json
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential

(x_train, y_train),(x_test,y_test) = fashion_mnist.load_data()


# for i in range (0,100):
#     im = Image.fromarray(x_train[i])
#     real = y_train[i]
#     im.save("./pic/%d_%d.jpeg" % (i,real))


input_shape = (28, 28, 1)
path = "./pic/"

val_count = 8000
x_val = x_test[:val_count]
y_val = y_test[:val_count]
x_test = x_test[val_count:]
y_test = y_test[val_count:]


x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_val = x_val.astype('float32') / 255.0

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))

test_images, test_labels = x_test.copy(), y_test.copy()

model = Sequential([
        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', 
               input_shape=(28,28,1)),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),        
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),    
        MaxPooling2D(pool_size=(2, 2)),   
        
        Flatten(),
        
        Dense(1024, activation='relu'),
        
        Dense(512, activation='relu'),
        
        Dense(10, activation='softmax')
    ])


adam = Adam(learning_rate=0.0001, decay=1e-6)
model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size = 1024, epochs = 100, verbose = 2 , validation_split = 0.1)

model.save("./models/model.mdl")
model.save_weights("./models/model.h5")

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()


model = keras.models.load_model("./models/model.mdl")
model.load_weights("./models/model.h5")

score = model.evaluate(x_test,y_test,verbose=2)
print("\n\nTest loss:", score[0])
print("Test accuracy:", score[1])


pred = model.predict(x_test)

print(pred[1]) #Prediction for image 1
pred_1 = np.argmax(pred[1])
print(pred_1)


"""
for i in range(0,100):
    pred_i = np.argmax(pred[i]) # get the position of the highest value within the list
    print (y_labels[i], pred_i)
"""
