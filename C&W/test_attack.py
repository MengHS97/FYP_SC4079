import base64
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
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel
from tabulate import tabulate
from l0_attack import CarliniL0

def show(img):
    """
    Show MNIST digits in the console.
    """
    remap = "  .*#" + "#" * 100
    img = (img.flatten() + 0.5) * 3
    if len(img) != 784:
        return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i * 28:i * 28 + 28]]))

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

def preprocess_image(img):
    """
    Apply preprocessing to the image before any other transformations.
    """
    mean = np.mean(img)
    std = np.std(img)
    preprocessed_image = (img - mean) / (std + 1e-8)
    return preprocessed_image

def scale_defense(img, scaling_factor=1.2):
    """
    Apply input scaling defense to an image.
    """
    preprocessed_image = preprocess_image(img)
    scaled_image = preprocessed_image * scaling_factor
    clipped_scaled_image = np.clip(scaled_image, 0, 1)
    return clipped_scaled_image

def generate_encryption_key(length=16):
    key = secrets.token_bytes(length)
    return key

def derive_parameters_from_key(encryption_key, num_parameters=3):
    salt = secrets.token_bytes(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        iterations=100000,
        salt=salt,
        length=32,
        backend=default_backend()
    )
    key = kdf.derive(encryption_key)

    parameters = {
        'key': key,
        'iv': secrets.token_bytes(16)
    }

    return parameters

def aes_encrypt(data, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    padded_data = pad(data, AES.block_size)
    cipher_text = cipher.encrypt(padded_data)
    return base64.b64encode(cipher_text), cipher.iv

def aes_decrypt(cipher_text, key, iv):
    cipher = AES.new(key, AES.MODE_CBC, iv)
    decrypted_data = cipher.decrypt(base64.b64decode(cipher_text))
    return unpad(decrypted_data, AES.block_size)

def encode_data(data, key, iv):
    flat_data = data.flatten()
    data_bytes = flat_data.tobytes()
    cipher_text, _ = aes_encrypt(data_bytes, key, iv)
    return cipher_text

def decode_data(encoded_data, key, iv, original_shape):
    decrypted_data_bytes = aes_decrypt(encoded_data, key, iv)
    decrypted_data = np.frombuffer(decrypted_data_bytes, dtype=np.float32).reshape(original_shape)
    return decrypted_data

def apply_custom_defense(img, defense_parameters):
    key = defense_parameters['key']
    iv = defense_parameters['iv']

    flat_img = img.flatten()
    img_bytes = flat_img.tobytes()

    encoded_data = encode_data(img, key, iv)

    # Now, let's add some random padding to the encoded data (more sophisticated defense)
    random_padding = secrets.token_bytes(AES.block_size - (len(encoded_data) % AES.block_size))
    defended_encoded_data = encoded_data + random_padding

    decoded_data = decode_data(defended_encoded_data, key, iv, img.shape)

    return decoded_data

def enhanced_gaussian_defense_with_denoising(img, encryption_key, epsilon_values, clip_range=(0, 1), random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    defense_parameters = derive_parameters_from_key(encryption_key)

    preprocessed_image = apply_custom_defense(img, defense_parameters)

    denoised_images = []

    for epsilon in epsilon_values:
        scaling_factor = np.random.uniform(0.5, 1.5)

        noise = np.random.normal(loc=0, scale=epsilon * scaling_factor, size=preprocessed_image.shape)

        noisy_image = np.clip(preprocessed_image + noise, clip_range[0], clip_range[1])

        denoised_image = gaussian_filter(noisy_image, sigma=1.0)

        denoised_images.append(denoised_image)

    return denoised_images

if __name__ == "__main__":
    encryption_key = generate_encryption_key()
    print("Encryption Key:", encryption_key)

    epsilon_values = [0.1, 0.15, 0.2]  # Example array of epsilon values

    defense_parameters = derive_parameters_from_key(encryption_key)
    print("Defense Parameters:", defense_parameters)

    parser = argparse.ArgumentParser()
    FLAGS = parser.parse_args()

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.disable_eager_execution()

        data = MNIST()
        model = MNISTModel("models/mnist.h5")

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
        inputs2, targets2 = generate_data(data, samples=samples, targeted=True, start=start, inception=isinstance(model, InceptionModel))

        adv_l0 = attack_l0.attack(inputs2, targets2)

        scaling_factor = 1.2
        scaled_adv_l0 = scale_defense(adv_l0, scaling_factor=scaling_factor)

        defended_adv_l0_list = []

        for epsilon in epsilon_values:
            defended_adv_l0 = enhanced_gaussian_defense_with_denoising(adv_l0, encryption_key=encryption_key, epsilon_values=epsilon_values, random_seed=random.randint(0, 1000))
            defended_adv_l0_list.append(defended_adv_l0[0])

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.model.compile(loss=tf.nn.softmax_cross_entropy_with_logits,
                    optimizer=sgd,
                    metrics=['accuracy'])

        weight = model.model.load_weights("models/mnist.h5")

        a = model.model.evaluate(inputs, targets, weight, batch_size)
        a_l0 = model.model.evaluate(adv_l0, targets2, weight, batch_size)
        a_l0_1 = model.model.evaluate(scaled_adv_l0, targets2, weight, batch_size)

        original_accuracy = ["Original Accuracy", "{:.2f}%".format(a[1] * 100)]
        adversarial_l0_accuracy = ["Adversarial Carlini L0 Accuracy", "{:.2f}%".format(a_l0[1] * 100)]
        adversarial_l0_accuracy_1 = ["Adversarial Carlini L0 Accuracy with Scaled", "{:.2f}%".format(a_l0_1[1] * 100)]

        adversarial_l0_accuracy_list = []

        for i, epsilon in enumerate(epsilon_values):
            a_l0_2 = model.model.evaluate(defended_adv_l0_list[i], targets2, weight, batch_size)
            adversarial_l0_accuracy_list.append(["Adversarial Carlini L0 Accuracy with AES, Epsilon = {}".format(epsilon), "{:.2f}%".format(a_l0_2[1] * 100)])

        table_data = [original_accuracy, adversarial_l0_accuracy, adversarial_l0_accuracy_1] + adversarial_l0_accuracy_list
        headers = ["Metric", "Accuracy"]
        table = tabulate(table_data, headers, tablefmt="grid")
        print(table)

        print("Evaluation complete.")
