import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import to_categorical

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, params, num_epochs=1, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3), input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10, activation='softmax'))  # Softmax activation for multi-class classification
    
    if init is not None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)
    
    if file_name is not None:
        # model.save(file_name)
        model.save_weights(file_name)

    return model

def train_distillation(data, file_name, params, num_epochs=1, batch_size=128, train_temp=1):
    if not os.path.exists(file_name + "_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name + "_init", params, 1, batch_size)

    # Now train the teacher at the given temperature
    teacher = train(data, file_name + "_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name + "_init")

    # Evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    y = tf.nn.softmax(predicted / train_temp, axis=-1).numpy()
    print("Softened labels shape:", y.shape)
    data.train_labels = y

    # Train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name + "_init")

    # Save the student model without including the optimizer states
    student.save(file_name, include_optimizer=False)

    # Finally, predict at temperature 1
    predicted = student.predict(data.train_data)
    print("Student model predictions shape:", predicted.shape)

if __name__ == "__main__":
    if not os.path.isdir('models'):
        os.makedirs('models')

    # Train standard models
    train(CIFAR(), "models/cifar.h5", [64, 64, 128, 128, 256, 256], num_epochs=3)
    train(MNIST(), "models/mnist.h5", [32, 32, 64, 64, 200, 200], num_epochs=3)

    # Distillation
    base_path = os.path.dirname(os.path.abspath(__file__))
    mnist_model_path = os.path.join(base_path, "models", "mnist-distilled-100")
    cifar_model_path = os.path.join(base_path, "models", "cifar-distilled-100")

    # train_distillation(MNIST(), mnist_model_path, [32, 32, 64, 64, 200, 200],
    #                    num_epochs=1, train_temp=100)
    # train_distillation(CIFAR(), cifar_model_path, [64, 64, 128, 128, 256, 256],
    #                    num_epochs=1, train_temp=100)