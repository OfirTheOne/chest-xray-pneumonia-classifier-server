from PIL import Image
import glob
import os
import cv2
import matplotlib.pyplot as plt

import numpy as np
import random

import tensorflow as tf
from tensorflow import keras

from sklearn import model_selection
from sklearn.metrics import accuracy_score

import itertools
import time


current_dir = os.path.dirname(os.path.abspath(__file__))
rel_path = "../model_resource/"
abs_model_resource_folder_path = os.path.abspath(os.path.join(current_dir, rel_path))

train_folder_path = os.path.join(abs_model_resource_folder_path, './' + '/pneumonia_dataset/train')
test_folder_path = os.path.join(abs_model_resource_folder_path, './' + '/pneumonia_dataset//test')

### const parameters ###

label_list = [
    'NORMAL',
    'PNEUMONIA'
]

label_indices = range(len(label_list))

### Preprocessing hiper parameter
IMAGE_DIMS = [100, 100]

BINARY_DETECTION_THRESHOLD = 0.65

KERNEL_SIZE = (3, 3)

PARAM_MATRIX = {
    'NODES_AMOUNT': [64],  # [32, 64, 128],
    'DANSE_LAYERS': [2],  # [0, 1, 2],
    'CONV_LAYERS': [1]  # [1, 2, 3],
}


### functions ###

def img_to_array(im):
    return np.asarray(im)


def shuffle_images_labels_with_order(im, lb):
    c = list(zip(im, lb))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b


def single_image_preprocess(image_path):
    im = cv2.imread(image_path)
    im = cv2.resize(im, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    im = img_to_array(im)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


def read_all_label_images(path_pattern):
    image_list = []
    for filename in glob.glob(path_pattern):
        im = single_image_preprocess(filename)
        image_list.append(im)

    image_list = np.array(image_list, dtype="float") / 255.0
    return image_list


def read_single_image_predict_and_display(model, label_list, filename, label):
    im = single_image_preprocess(filename)
    predictions = model.predict([[im]])

    plt.grid(False)
    plt.imshow(im, cmap=plt.cm.binary)
    plt.xlabel("Actual: " + label_list[label])
    plt.title("Prediction: " + label_list[np.argmax(predictions[0])])
    plt.show()


def read_labeled_images(path_to_lables_folder):
    image_label_pair_list = []
    i = 0
    for label in label_list:
        print("start reading from : " + path_to_lables_folder + '/' + label + '/**')
        image_list = read_all_label_images(path_to_lables_folder + '/' + label + '/**')
        print("done")

        image_list = map(lambda im: [label_indices[i], im], image_list)
        image_list = np.asarray(list(image_list))
        image_label_pair_list = np.concatenate([image_label_pair_list, image_list]) if len(
            image_label_pair_list) else image_list
        i = i + 1

    [labels, images] = map(lambda x: np.asarray(list(x)), zip(*image_label_pair_list))
    return labels, images


def param_matrix_to_contexts_list(params_matrix):
    fields = params_matrix.keys()
    values_list = list(map(lambda f: params_matrix[f], fields))
    permutation_list = list(itertools.product(*values_list))
    value_group_list = list(map(lambda val_list: dict(zip(fields, val_list)), permutation_list))
    return value_group_list


def serialize_model(model, model_name):
    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(model_name+".h5")
    print("Saved model to disk")


def load_model(model_path, weights_path):
    model_json_string = open(model_path).read()
    restored_model = keras.models.model_from_json(model_json_string)
    restored_model.load_weights(weights_path)
    return restored_model


def build_model_by_hyper_params(CONV_LAYERS, NODES_AMOUNT, DANSE_LAYERS):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            NODES_AMOUNT,
            input_shape=(IMAGE_DIMS[0], IMAGE_DIMS[1], 1),
            data_format="channels_last",
            kernel_size=KERNEL_SIZE,
            activation="relu"
        )
    )

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    for i in (range(CONV_LAYERS - 1) if CONV_LAYERS > 0 else []):
        model.add(
            keras.layers.Conv2D(
                NODES_AMOUNT,
                data_format="channels_last",
                kernel_size=KERNEL_SIZE,
                activation="relu"
            )
        )
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(NODES_AMOUNT, activation="relu"))

    for i in (range(DANSE_LAYERS - 1) if CONV_LAYERS > 0 else []):
        model.add(keras.layers.Dense(NODES_AMOUNT, activation="relu"))
        model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        optimizer= 'rmsprop', #'adam',
        loss='binary_crossentropy',  # 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def plot_loss_data(history):
    plt.plot(history['loss'], label='MAE (testing data)')
    plt.plot(history['val_loss'], label='MAE (validation data)')
    plt.title('MAE for Chennai Reservoir Levels')
    plt.ylabel('MAE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()


def plot_labels_histogram(labels_list) :
    (unique, counts) = np.unique(labels_list, return_counts=True)
    label_counts = counts
    fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    ax1.bar(np.arange(len(label_counts)) + 0.5, label_counts)
    ax1.set_xticks(np.arange(len(label_counts)) + 0.5)
    _ = ax1.set_xticklabels(unique)
    plt.show()


def main():
    train_labels, train_images = read_labeled_images(train_folder_path)
    train_labels, train_images = shuffle_images_labels_with_order(train_labels, train_images)

    plot_labels_histogram(train_labels)

    test_labels, test_images = read_labeled_images(test_folder_path)
    test_labels, test_images = shuffle_images_labels_with_order(test_labels, test_images)

    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)

    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    print('finish preprocess.')

    ### build model ###

    contexts_list = param_matrix_to_contexts_list(PARAM_MATRIX)

    for context in contexts_list:
        DANSE_LAYERS = context['DANSE_LAYERS']
        NODES_AMOUNT = context['NODES_AMOUNT']
        CONV_LAYERS = context['CONV_LAYERS']

        MODEL_NAME = '{}-conv--{}-nodes--{}-dense--{}'.format(CONV_LAYERS, NODES_AMOUNT, DANSE_LAYERS, int(time.time()))
        LOG_DIR = 'logs/{}'.format(MODEL_NAME)
        print(MODEL_NAME)

        model = build_model_by_hyper_params(CONV_LAYERS, NODES_AMOUNT, DANSE_LAYERS)

        model.summary()

        tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
        history = model.fit(
            np.expand_dims(train_images, axis=-1),
            train_labels,
            epochs=5,
            batch_size=64,
            validation_split=0.2,
            callbacks=[tensorboard_callback]
        )

        plot_loss_data(history.history)

        model.evaluate(np.expand_dims(test_images, axis=-1), np.expand_dims(test_labels, axis=-1))

        predictions = model.predict(np.expand_dims(test_images, axis=-1))
        rounded_predictions = np.asarray(list(map(lambda p: 1 if p > BINARY_DETECTION_THRESHOLD else 0, predictions)))
        predictions_labels_indices = rounded_predictions

        print('accuracy_score: {}'.format(accuracy_score(predictions_labels_indices, test_labels)))

        serialize_model(model, MODEL_NAME)

        loaded_model = load_model(MODEL_NAME+'.json', MODEL_NAME+'.h5')
        predictions_2 = loaded_model.predict(np.expand_dims(test_images, axis=-1))
        rounded_predictions_2 = np.asarray(list(map(lambda p: 1 if p > BINARY_DETECTION_THRESHOLD else 0, predictions_2)))
        predictions_labels_indices_2 = rounded_predictions_2

        print('after restore - accuracy_score: {}'.format(accuracy_score(predictions_labels_indices_2, test_labels)))

main()