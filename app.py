import flask
import os
import pickle
import pandas as pd
import numpy as np
import skimage
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
import create_model
from werkzeug.utils import secure_filename

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = flask.Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

gender_classifier_path = 'models/gender_model.pkl'

# load models here with pickle from models folder, if there is an error, then generate models
try:
    model_file = open(gender_classifier_path, 'rb')
    gender_classifier = pickle.load(model_file)
except IOError:
    print("Error finding gender classifier model, recreating")
    gender_classifier = create_model.create_gender_model()  # returns model for gender and also saves it to models folder

# load age model here


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            filename = secure_filename(file.filename)
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_image_path)  # saving original image before we resize, to display later
            print(original_image_path)

            img = skimage.io.imread(original_image_path)  # Read the image using skimage
            img = skimage.color.rgb2gray(img)  # convert to grayscale
            img = skimage.transform.resize(image=img, output_shape=(48, 48))  # downsize to size of our dataset
            mod_image_path = (os.path.splitext(original_image_path)[0] + '_downscaled'
                              + os.path.splitext(original_image_path)[1])
            skimage.io.imsave(mod_image_path, img)  # save our modified file that we use with our model
            
            img = img.ravel()  # flatten 48*48 array to 1x(48*48)
            img = img * 255  # our model expects int value for each pixel 0 - 255
            img = img.astype(int)
            print(img, img.shape, type(img))

            gender_prediction = gender_classifier.predict([img])
            print(gender_prediction)

            if gender_prediction[0] == 0:
                gender_prediction = 'Male'
            elif gender_prediction[0] == 1:
                gender_prediction = 'Female'
            else:
                gender_prediction = None  # error

            # age =

            newpath = ""
            for char in original_image_path:
                if char == '\\':
                    newpath += '/'
                else:
                    newpath += char
            original_image_path = 'static' + newpath.split('static')[1]  # only keep path after 'static'
            print(original_image_path)
            newpath = ""
            for char in mod_image_path:
                if char == '\\':
                    newpath += '/'
                else:
                    newpath += char
            mod_image_path = 'static' + newpath.split('static')[1]  # only keep path after 'static'
            print(mod_image_path)

            return flask.render_template('classify_image.html', gender_prediction=gender_prediction,
                                         original_image=original_image_path, new_image=mod_image_path)
        else:
            pass  # display that there was error getting file

    return flask.render_template('main.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    # results page
    return flask.render_template('classify_image.html')


if __name__ == '__main__':
    app.run(debug=True)
