import flask
import os
import pickle
import pandas as pd
import skimage

app = flask.Flask(__name__, template_folder='templates')

gender_classifier_path = 'models/gender_model.pkl'

# load models here with pickle from models folder, if there is an error, then generate models
try:
    file = open(gender_classifier_path, 'rb')
    gender_classifier = pickle.load(file)
except IOError:
    print("Error finding gender classifier model")
    # recreate model here


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file:
            # Read the image using skimage
            img = skimage.io.imread(file)
            img = skimage.transform.resize(img, (28, 28))
            img = img.flatten()

            # make predictions here
            # gender =
            # age =

            # Get the value of the prediction

            return flask.render_template('classify_image.html', prediction=str(prediction))
        else:
            pass # display that there was error getting file

    return flask.render_template('main.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    # results page after image found and face found
    return flask.render_template('classify_image.html')


if __name__ == '__main__':
    app.run(debug=True)
