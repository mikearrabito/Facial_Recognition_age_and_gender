import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import reformat_csv


try:
    df = pd.read_csv('data/age_gender_updated.csv')
except IOError:
    try:
        df = pd.read_csv('data/age_gender.csv')
        print("Original dataset found, reformatting")
        reformat_csv.pixels_to_columns('data/age_gender.csv')
    except IOError:
        print("Error finding age_gender.csv file, unable to create dataframe")


def create_gender_model():
    list_of_pixel_cols = list()
    for i in range(48 * 48):
        list_of_pixel_cols.append('pixel' + str(i))

    if df['pixels']: # if we have pixels column from our original dataset
        reformat_csv.pixels_to_columns('data/age_gender_updated.csv')

    X = df[list_of_pixel_cols]  # X is the 48x48 image
    y = df['gender']  # 0 - man, 1 - woman

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)
    svc_model = svm.SVC()
    svc_model.fit(X_train, y_train)
    accuracy = svc_model.score(X_test, y_test)
    print("Accuracy for SVC: %f" % accuracy)

    filepath = 'models/gender_model.pkl'
    pickle.dump(svc_model, open(filepath, 'wb'))  # save model to our models folder
    return svc_model


def create_age_model():
    #

    return


'''  
x_train = np.divide(X_train.values.astype(float), 255) #divides all values by 255, so all of our values are now between 0 and 1
x_test = np.divide(X_test.values.astype(float), 255)
#print(x_train, x_test)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

sz_train = x_train.shape[0] # number of rows we have
sz_test = x_test.shape[0]
x_train = x_train.reshape(sz_train, 48, 48, -1) # converts to 48x48 center columns, numpy determines last column
x_test = x_test.reshape(sz_test, 48, 48, -1) # reshape x_test like above

y_train = np.array(y_train) 
y_test = np.array(y_test)

#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
# shape should be (15882, 48, 48, 1) (7823, 48, 48, 1) (15882,) (7823,)

tf_model = tf.keras.models.Sequential()
tf_model.add(tf.keras.layers.Conv2D(input_shape=(48,48,1),filters=28,kernel_size=(3,3), strides=(1,1), activation="relu"))
tf_model.add(tf.keras.layers.Conv2D(filters=28,kernel_size=(3,3), strides=(1,1), activation="relu"))


tf_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['acc'])
tf_model.fit(x_train, y_train, epochs=10, validation_split=0.1)
tf_model.evaluate(x_test, y_test)

# tf_model.save('models/tf_model') 
# tf_model = keras.models.load_model('models/tf_model') #for loading nn
'''