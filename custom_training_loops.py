import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import time

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

def train(model, training_data, loss_function, optimizer, epochs):
    model.compile(loss=loss_function, optimizer=optimizer, metrics=["accuracy"])

    for epoch in range(epochs):
        for step, (x_train_batch, y_train_batch) in enumerate(training_data):
            with tf.GradientTape() as tape:
                logits = model(x_train_batch, training=True)
                loss_value = loss_fn(y_train_batch, logits)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            if step % 20 == 0:
                print("Training Loss at Epoch " + str(epoch) + " Step " + str(step) + ":" + str(float(loss_value)))
    return model

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 64
epochs = 15

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data for training
x_train, y_train = preprocess_data(x_train, y_train)
x_test, y_test = preprocess_data(x_test, y_test)

input_layer, encoder_output = encoder(input_shape)
softmax_output = prediction_layer(encoder_output, num_classes)

model = Model(input_layer, softmax_output)
model.summary()

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
val_dataset = val_dataset.batch(batch_size)

loss_fn = keras.losses.CategoricalCrossentropy()
optimizer = keras.optimizers.SGD(learning_rate=1e-3)


model = train(model=model, training_data=train_dataset,
              loss_function=loss_fn, optimizer=optimizer,
              batch_size=batch_size, epochs=epochs)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


