## verify.py -- check the accuracy of a neural network
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licensed under the BSD 2-Clause license,
## contained in the LICENSE file in this directory.

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

import tensorflow as tf
import numpy as np

BATCH_SIZE = 1

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main():
    with tf.compat.v1.Session() as sess:
        data, model = MNIST(), MNISTModel("models/mnist", sess)
        data, model = CIFAR(), CIFARModel("models/cifar", sess)
        # data, model = ImageNet(), InceptionModel(sess)

        x = tf.compat.v1.placeholder(tf.float32, (None, model.image_size, model.image_size, model.num_channels))
        y = model.predict(x)

        r = []
        for i in range(0, len(data.test_data), BATCH_SIZE):
            pred = sess.run(y, feed_dict={x: data.test_data[i:i + BATCH_SIZE]})
            r.append(np.argmax(pred, axis=1) == np.argmax(data.test_labels[i:i + BATCH_SIZE], axis=1))
            print("Accuracy so far:", np.mean(r))
    
if __name__ == "__main__":
    main()