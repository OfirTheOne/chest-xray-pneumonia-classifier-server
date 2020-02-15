import setting
# --
import os
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from model.predict import predict_by_image_url, predict_by_image_file, ndarray_to_list

from utils.index import rel_to_abs

UPLOAD_FOLDER = '/upload_files'

app = Flask(__name__, template_folder="views")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.after_request
def apply_caching(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/resource/<label>/<name>')
def send_resource(label, name):
    return send_from_directory(os.path.join(rel_to_abs(__file__, './model_resource/pneumonia_dataset/val'), label), name)


@app.route('/available_resource')
def get_available_resource():

    normal_path = rel_to_abs(__file__, "./model_resource/pneumonia_dataset/val/NORMAL/")
    pneumonia_path = rel_to_abs(__file__, "./model_resource/pneumonia_dataset/val/PNEUMONIA/")

    normal_sample = os.listdir(normal_path)
    pneumonia_sample = os.listdir(pneumonia_path)

    return jsonify(NORMAL=normal_sample, PNEUMONIA=pneumonia_sample)


@app.route('/')
def hello():
    return jsonify(data=[os.getenv('MEANING_OF_LIFE')])


@app.route('/predict')
def predict():
    image_url = request.args.get('image_url')
    probability_value, label = predict_by_image_url(image_url)
    return jsonify(prediction={'probability_value': str(probability_value), 'label': label})


@app.route('/predict/view')
def predict_view():
    image_url = request.args.get('image_url')
    pred, label = predict_by_image_url(image_url)

    template_context = dict(
        image_url=image_url,
        prediction=label,
        prediction_prob=pred
    )

    return render_template('prediction-result.html', **template_context)


@app.route('/predict/upload', methods=['POST'])
def predict_upload_file():
    image_field_name = 'image'
    try:
        if request.method == 'POST':

            if image_field_name not in request.files:
                raise Exception('No file part')

            file = request.files[image_field_name]
            if file.filename == '':
                raise Exception('No selected file')

            if file:
                probability_value, label = predict_by_image_file(file)
                return jsonify(prediction={'probability_value': str(probability_value), 'label': label})
            else:
                raise Exception('Process failed')

    except Exception as err:
        flash(err)
        return redirect(request.url)


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    app.debug = True
    app.run()
