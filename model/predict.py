from model.load_model import model

import cv2
import numpy as np
from io import BytesIO
import requests
from PIL import Image

import itertools

IMAGE_DIMS = [100, 100]
BINARY_DETECTION_THRESHOLD = 0.65

label_list = [
    'NORMAL',
    'PNEUMONIA'
]


# region - main prediction procedures
def predict_by_image_url(image_url):
    image = download_image(image_url)
    image = preprocess_image(image)
    probability_value, label = predict_processed_image(image)
    return probability_value, label


def predict_by_image_file(image_file):
    image = file_to_image(image_file)
    image = preprocess_image(image)
    probability_value, label = predict_processed_image(image)
    return probability_value, label


def predict_processed_image(image):
    predictions = model.predict(np.expand_dims([image], axis=-1))
    predictions_rounded = round_binary_predictions(predictions, BINARY_DETECTION_THRESHOLD)

    label = prediction_to_label(predictions_rounded[0])
    probability_value = np.asarray(predictions).flatten()[0]

    return probability_value, label
# endregion


# region - file source handling
def download_image(image_url):
    data = requests.get(image_url).content
    return raw_image_bytes_to_image_object(data)


def file_to_image(file):
    data = file.read()
    image = raw_image_bytes_to_image_object(data)
    return image


def raw_image_bytes_to_image_object(data):
    return Image.open(BytesIO(data))
# endregion


# region- preprocess
def preprocess_image(raw_image):
    image_matrix = image_to_array(raw_image)
    resized_image = cv2.resize(image_matrix, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    grayscale_image = convert_image_to_grayscale(resized_image) if len(resized_image.shape) > 2 else resized_image
    normalize_image = normalize_pixel_array(grayscale_image)
    return normalize_image


def image_to_array(image):
    return np.asarray(image)


def convert_image_to_grayscale(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# endregion


# region - prediction transform
def normalize_pixel_array(pixel_array):
    return np.array(pixel_array, dtype="float") / 255.0


def prediction_to_label(single_prediction_list):
    return label_list[single_prediction_list]


def round_binary_predictions(predictions, threshold):
    return np.asarray(list(map(lambda p: 1 if p > threshold else 0, predictions)))
# endregion


# region - utility
def ndarray_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
# endregion

