import tensorflow as tf
import numpy as np
import time
import random
import argparse
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from keras.optimizers import SGD
import cv2
import secrets
from setup_cifar import CIFAR, CIFARModel
from tabulate import tabulate
from l0_attack import CarliniL0

def show(img):
    """
    Show CIFAR-10 images in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + 0.5) * 3
    if len(img) != 3072:
        return
    print("START")
    for i in range(32):
        print("".join([remap[int(round(x))] for x in img[i * 32 : i * 32 + 32]]))

def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data for the attack algorithm.
    """
    inputs = []
    targets = []

    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(data.test_labels.shape[1]), 10)
            else:
                seq = [j for j in range(data.test_labels.shape[1]) if j != np.argmax(data.test_labels[start + i])]

            for j in seq:
                inputs.append(data.test_data[start + i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start + i])
            targets.append(data.test_labels[start + i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets

def enhanced_gaussian_defense_with_denoising(img, epsilon_values, clip_range=(0, 1), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    denoised_images = []

    for epsilon in epsilon_values:
        scaling_factor = np.random.uniform(0.5, 1.5)

        noise = np.random.normal(loc=0, scale=epsilon * scaling_factor, size=img.shape)

        noisy_image = np.clip(img + noise, clip_range[0], clip_range[1])

        denoised_image = gaussian_filter(noisy_image, sigma=1.0)

        denoised_images.append(denoised_image)

    return denoised_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS = parser.parse_args()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.disable_eager_execution()

        data = CIFAR()
        model = CIFARModel("models/cifar.h5")

        epsilon_values = [0.1, 0.15, 0.2]  # Example array of epsilon values

        accuracies = []

        batch_size = 32
        samples = 20
        max_iterations = 1000
        confidence = 0

        initial_const = random.randint(0, 100)
        largest_const = random.randint(0, 100)

        if initial_const > largest_const:
            initial_const, largest_const = largest_const, initial_const

        attack_l0 = CarliniL0(sess, model, max_iterations=max_iterations, initial_const=10, largest_const=15)

        start = 1
        inputs, targets = generate_data(data, samples=samples, targeted=False)
        inputs2, targets2 = generate_data(data, samples=samples, targeted=True, start=start, inception=False)

        adv_l0 = attack_l0.attack(inputs2, targets2)

        scaling_factor = 1.2
        scaled_adv_l0 = adv_l0  # No scaling in this case

        defended_adv_l0_list = []

        for epsilon in epsilon_values:
            defended_adv_l0 = enhanced_gaussian_defense_with_denoising(adv_l0, epsilon_values=epsilon_values, random_seed=random.randint(0, 1000))
            defended_adv_l0_list.append(defended_adv_l0[0])

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.model.compile(loss=tf.nn.softmax_cross_entropy_with_logits,
                    optimizer=sgd,
                    metrics=['accuracy'])

        weight = model.model.load_weights("models/cifar.h5")

        a = model.model.evaluate(inputs, targets, weight, batch_size)
        a_l0 = model.model.evaluate(adv_l0, targets2, weight, batch_size)
        a_l0_1 = model.model.evaluate(scaled_adv_l0, targets2, weight, batch_size)

        original_accuracy = ["Original Accuracy", "{:.2f}%".format(a[1] * 100)]
        adversarial_l0_accuracy = ["Adversarial Carlini L0 Accuracy", "{:.2f}%".format(a_l0[1] * 100)]
        adversarial_l0_accuracy_1 = ["Adversarial Carlini L0 Accuracy with Scaled", "{:.2f}%".format(a_l0_1[1] * 100)]

        adversarial_l0_accuracy_list = []

        for i, epsilon in enumerate(epsilon_values):
            a_l0_2 = model.model.evaluate(defended_adv_l0_list[i], targets2, weight, batch_size)
            adversarial_l0_accuracy_list.append(["Adversarial Carlini L0 Accuracy with Gaussian Defense, Epsilon = {}".format(epsilon), "{:.2f}%".format(a_l0_2[1] * 100)])

        table_data = [original_accuracy, adversarial_l0_accuracy, adversarial_l0_accuracy_1] + adversarial_l0_accuracy_list
        headers = ["Metric", "Accuracy"]
        table = tabulate(table_data, headers, tablefmt="grid")
        print(table)

        print("Evaluation complete.")
