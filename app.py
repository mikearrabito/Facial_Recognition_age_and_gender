import flask
import os
import pickle
import pandas as pd
import skimage

app = flask.Flask(__name__, template_folder='templates')

path_to_image_classifier = 'models/image-classifier.pkl'

# add our model here
with open(path_to_image_classifier, 'rb') as f:
    image_classifier = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')

    if flask.request.method == 'POST':
        # Get file object from user input.
        file = flask.request.files['file']

        if file: # (if file and face found in picture)
            # Read the image using skimage
            img = skimage.io.imread(file)

            # Resize the image to match the input the model will accept
            img = skimage.transform.resize(img, (28, 28))

            # Flatten the pixels from 28x28 to 784x0
            img = img.flatten()

            # Get prediction of image from classifier
            predictions = image_classifier.predict([img])

            # Get the value of the prediction
            prediction = predictions[0]

            return flask.render_template('classify_image.html', prediction=str(prediction))
        else:
            pass # display that there was error getting file or error finding face in picture

    return flask.render_template('main.html')


@app.route('/classify_image/', methods=['GET', 'POST'])
def classify_image():
    # results page after image found and face found
    return flask.render_template('classify_image.html')


if __name__ == '__main__':
    app.run(debug=True)
