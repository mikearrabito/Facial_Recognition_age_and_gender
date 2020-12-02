import pandas as pd
import pickle
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import train_test_split
import reformat_csv
import tensorflow as tf
from tensorflow import keras

try:
    df = pd.read_csv('data/age_gender_updated.csv')
except IOError:
    try:
        df = pd.read_csv('data/age_gender.csv')
        print("Original dataset found, reformatting")
        reformat_csv.pixels_to_columns('data/age_gender.csv')
    except IOError:
        print("Error finding age_gender.csv file, unable to create dataframe")

list_of_pixel_cols = list()
for i in range(48 * 48):
    list_of_pixel_cols.append('pixel' + str(i))


def create_gender_model():
    if 'pixels' in df.columns:  # if we have pixels column from our original dataset
        reformat_csv.pixels_to_columns('data/age_gender.csv')

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


def create_age_model_sk():
    df_ages = pd.DataFrame(df)
    df_ages['age_class'] = df['age'] / 10
    df_ages = df.astype(int)

    X = df_ages[list_of_pixel_cols]  # X is the 48x48 image
    y = df_ages['age_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)
    
    model = svm.SVC()
    model.fit(X_train, y_train)
    filepath = 'models/age_model.pkl'
    pickle.dump(model, open(filepath, 'wb'))  # save model to our models folder
    return model


def create_age_model_tf():
    df_ages = pd.DataFrame(df)
    df_ages['age_class'] = df['age'] / 10
    df_ages = df_ages.astype(int)

    X = df_ages[list_of_pixel_cols]  # X is the 48x48 image
    y = df_ages['age_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=45)

    # divide all values by 255, so all of our values are now between 0 and 1
    x_train = np.divide(X_train.values.astype(float), 255)
    x_test = np.divide(X_test.values.astype(float), 255)
    # print(x_train, x_test)
    # print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    sz_train = x_train.shape[0]  # number of rows we have
    sz_test = x_test.shape[0]
    x_train = x_train.reshape(sz_train, 48, 48, -1)  # converts to 48x48 center columns
    x_test = x_test.reshape(sz_test, 48, 48, -1)  # reshape x_test like above

    y_train = np.array(y_train)
    y_test = np.array(y_test)

    model_age_class = keras.models.Sequential([
        keras.layers.Conv2D(64, 7, activation="relu",
                            input_shape=[48, 48, 1]),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 3, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(256, 3, activation="relu"),
        keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(12, activation="softmax")
    ])

    model_age_class.compile(optimizer="Nadam", loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

    model_age_class.fit(x_train, y_train, epochs=20,
                        callbacks=[early_stopping_cb],
                        batch_size=64,
                        validation_split=0.2)

    model_age_class.save('models/age_model/tf_age_model.h5')
    return model_age_class


def get_age_range(age):
    range = ""
    if age == 0:
        range = "0-9"
    elif age == 1:
        range = "10-19"
    elif age == 2:
        range = "20-29"
    elif age == 3:
        range = "30-39"
    elif age == 4:
        range = "40-49"
    elif age == 5:
        range = "50-59"
    elif age == 6:
        range = "60-69"
    else:
        range = "70+"
    return range


