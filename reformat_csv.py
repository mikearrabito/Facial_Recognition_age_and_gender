import pandas as pd
import numpy as np

# this files purpose is to reformat the original csv with pixel column being a string of ints to
# individual columns for each pixel and removing columns we do not need


def pixels_to_columns(path):
    if len(path) < 5 or path[-4:] != '.csv':
        print("CSV file expected, invalid file detected")
        return

    df = pd.read_csv(path)
    list_of_pixel_cols = list()  # generating columns for each pixel
    for i in range(48 * 48):
        list_of_pixel_cols.append('pixel' + str(i))

    images = df['pixels']
    images = images.str.split(' ')  # pixels column is now a list

    pixels_df = pd.DataFrame(images)
    pixels_df[list_of_pixel_cols] = pd.DataFrame(pixels_df.pixels.tolist(),
                                                 index=pixels_df.index)  # move each pixel to its own column
    pixels_df = pixels_df.drop('pixels', axis=1)  # get rid of original pixels column
    # print(pixels_df)

    df2 = pd.DataFrame(df)
    df2 = pd.concat([df2, pixels_df], axis=1, join='outer')  # concat with original df
    df2 = df2.drop(['pixels', 'img_name'], axis=1)  # drop columns that are not needed
    print(df2)

    df2.to_csv((path - '.csv') + '_updated.csv')  # save our modified csv file
    return
