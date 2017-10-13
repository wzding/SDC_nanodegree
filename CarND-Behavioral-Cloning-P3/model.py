# using keras 1.2.2
import tensorflow as tf
import keras.backend.tensorflow_backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
import csv
import cv2


def get_all_lines(path):
    """
    get all lines in a .csv file
    """
    lines = []
    with open(path) as file:
        reader = csv.reader(file)
        # skip header
        next(file, None)
        for line in reader:
            lines.append(line)
    return lines


def get_all_images(current_path, lines):
    """
    obtain all image paths and steering angle
    """
    images = []
    steering_angles = []
    for line in lines :
        # read in images from center, left and right cameras
        img_center = cv2.imread(current_path + line[0].split("/")[-1])
        img_left = cv2.imread(current_path + line[1].split("/")[-1])
        img_right = cv2.imread(current_path + line[2].split("/")[-1])
        if not any(i is None for i in [img_center, img_left, img_right]):
            # steering
            steering_center = float(line[3])
            # create adjusted steering measurements for the side camera images
            correction = 0.2 # this is a parameter to tune
            steering_left = steering_center + correction
            steering_right = steering_center - correction
            # add images and angles to data set
            images.extend([img_center, img_left, img_right])
            steering_angles.extend([steering_center, steering_left, steering_right])
    return list(zip(images, steering_angles))


def generator(samples, batch_size=32):
    """
    Args:
        samples(list): a list of tuples
    """

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            # create one dataset for X, one for y
            images = []
            angles = []
            for img, agl in batch_samples:
                # flip image
                flipped = cv2.flip(img, 1)
                images.extend([img, flipped])
                angles.extend([agl, agl * -1.0])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


def revised_NVidia():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50, 20), (0, 0))))
    model.add(Convolution2D(nb_filter=24, nb_row=5, nb_col=5, subsample=(2,2), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Convolution2D(nb_filter=36, nb_row=5, nb_col=5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(nb_filter=48, nb_row=5, nb_col=5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation="relu"))
    model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))
    return model


def main():
    lines = get_all_lines('./data/driving_log.csv')
    samples = get_all_images("./data/IMG/", lines)
    print(len(samples))
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)
    sess = tf.Session()
    K.set_session(sess)
    with tf.device('/gpu:0'):
        model = revised_NVidia()
        model.compile(loss="mse", optimizer="adam")
        history_object = model.fit_generator(
            train_generator,
            samples_per_epoch=len(train_samples),
            validation_data=validation_generator,
            nb_val_samples=len(validation_samples),
            nb_epoch=10, verbose=1)
        model.save('./model.h5')
    # fig = plt.figure(1, figsize=(8, 6))
    # plt.plot(history_object.history['loss'])
    # plt.plot(history_object.history['val_loss'])
    # plt.title('model mean squared error loss')
    # plt.ylabel('mean squared error loss')
    # plt.xlabel('epoch')
    # plt.legend(['training set', 'validation set'], loc='upper right')
    # fig.savefig('mse_plot.png')


if __name__ == "__main__":
    main()
