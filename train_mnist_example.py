import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, Model

def preprocess_data(x, y):
    x = x.astype("float32") / 255
    x = np.expand_dims(x, -1)
    y = keras.utils.to_categorical(y, num_classes)
    return x, y

def encoder(input_shape):
    input_layer = keras.Input(shape=input_shape)
    conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")(input_layer)
    maxpool1 = layers.MaxPool2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")(maxpool1)
    maxpool2 = layers.MaxPool2D(pool_size=(2, 2))(conv2)

    return input_layer, maxpool2

def prediction_layer(max_pool2, num_classes):
    flattened_layer = layers.Flatten()(max_pool2)
    dropout = layers.Dropout(0.5)(flattened_layer)
    classification_layer = layers.Dense(num_classes, activation="softmax")(dropout)
    return classification_layer

def train(model, training_data, batch_size, epochs):
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(training_data[0], training_data[1],
              batch_size=batch_size, epochs=epochs, validation_split=0.1)
    return model

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data for training
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

input_layer, encoder_output = encoder(input_shape)
softmax_output = prediction_layer(encoder_output, num_classes)

model = Model(input_layer, softmax_output)
model.summary()

model = train(model=model, training_data=(x_train, y_train), batch_size=128, epochs=15)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


