import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import accuracy_score


model_file_name = os.getenv('MODEL_FILE_NAME')
weights_file_name = os.getenv('WEIGHTS_FILE_NAME')

current_dir = os.path.dirname(os.path.abspath(__file__))
rel_path = "../model_resource/"
abs_model_folder_path = os.path.abspath(os.path.join(current_dir, rel_path))

model_json_string = open(os.path.join(abs_model_folder_path, './'+model_file_name)).read()
model = keras.models.model_from_json(model_json_string)
model.load_weights(os.path.join(abs_model_folder_path, './'+weights_file_name))



