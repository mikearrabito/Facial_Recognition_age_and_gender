import flask
import os
import pickle
import pandas as pd
import numpy as np
import skimage
from skimage import io, color
from skimage.transform import rescale, resize, downscale_local_mean
from werkzeug.utils import secure_filename
from face_detection import Face, find_faces
from create_model import create_gender_model


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app = flask.Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# clear faces and uploads folder here


gender_classifier_path = 'models/gender_model.pkl'

# load models here with pickle from models folder, if there is an error, then generate models
try:
    model_file = open(gender_classifier_path, 'rb')
    gender_classifier = pickle.load(model_file)
except IOError:
    print("Error finding gender classifier model, recreating")
    gender_classifier = create_gender_model()  # returns model for gender and saves it to models folder

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
            #print(original_image_path)

            images_of_faces = find_faces(original_image_path)  # returns a list of Faces
            gender_prediction = None
            mod_image_path = None

            if len(images_of_faces) > 0:
                for face in images_of_faces:
                    img = skimage.io.imread(face.image_path)
                    img = skimage.color.rgb2gray(img)  # convert to grayscale
                    img = skimage.transform.resize(image=img, output_shape=(48, 48))  # downsize to size of our dataset
                    mod_image_path = (os.path.splitext(original_image_path)[0] + '_downscaled'
                                      + os.path.splitext(original_image_path)[1])
                    skimage.io.imsave(mod_image_path, img)  # save our modified file that we use with our model
                    img = img.ravel()  # flatten 48*48 array to 1x(48*48)
                    img = img * 255  # our model expects int value for each pixel 0 - 255
                    img = img.astype(int)

                    gender_prediction = gender_classifier.predict([img])
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
                    newpath = ""
                    for char in mod_image_path:
                        if char == '\\':
                            newpath += '/'
                        else:
                            newpath += char
                    mod_image_path = 'static' + newpath.split('static')[1]  # only keep path after 'static'

                    break  #  remove this

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
