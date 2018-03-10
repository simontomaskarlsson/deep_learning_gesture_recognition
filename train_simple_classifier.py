from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import datetime
import tensorflow as tf

import os
import glob
import numpy as np
from time import sleep
from PIL import Image, ImageDraw

sys.path.insert(0, './models')
sys.path.insert(0, './misc')

from classify_webcam import startWebcam

import direction_gesture_data as direction_gesture_data
import help_gesture_data as help_gesture_data
import IR_data as IR_data
import simple_conv
import simple_conv_IR
import env

def run_test_data(images, labels, session, x, keep_prob, y_conv, y_, accuracy,
                    DEBUG = False, WEBCAM_TEST = False):
    if not DEBUG:
        print('Test accuracy: %g' % accuracy.eval(feed_dict={
            x: images, y_: labels, keep_prob: 1.0}))
    else:

        files = glob.glob('data/incorrectClassifications/*')

        for f in files:
            os.remove(f)

        # Select the correct class index from each label-array
        labels = [np.argmax(x) for x in labels]

        # Run prediction
        predictionsNumpy = session.run(y_conv, feed_dict={x: images, keep_prob: 1})

        # Select the correct class index from each prediction-array
        predictions = [np.argmax(x) for x in predictionsNumpy]

        print("Test accuracy:", sum([labels[myIndex] == predictions[myIndex] for myIndex in range(len(images))])/len(images))

        # Draw image section
        for myIndex in range(len(images)):
            if labels[myIndex] != predictions[myIndex]:
                imageArray = np.copy(images[myIndex])

                # image size limit
                limit = int((len(imageArray))**(1/2))

                # Change image format from tensorflow preferences to PIL preferences
                imageArrayFormatted = np.empty((limit,limit,3), dtype=np.uint8)
                for y in range(0, limit):
                    for x_ in range(0, limit):
                        imageArrayFormatted[y][x_] = [int(pixel) for pixel in imageArray[y*limit + x_]]
                im = Image.fromarray(imageArrayFormatted, "RGB")

                middle = limit/2
                draw = ImageDraw.Draw(im)

                # (x,y)-format
                classes = [(middle,middle), (41,0), (limit,0), (limit, 21), (limit, 40), (limit, limit), (40,limit), (21,limit), (0,limit), (0,40), (0, 21),(0,0), (24,0)]

                draw.line(((middle, middle), classes[labels[myIndex]]), fill=(250, 0, 0))
                draw.line(((middle, middle), classes[predictions[myIndex]]), fill=(0, 0, 250))

                im.save("./data/incorrectClassifications/incorrectImage" + str(myIndex) + ".png")
    if WEBCAM_TEST:
        # cam here
        print("starting webcam")
        startWebcam(session, x, y_conv, keep_prob)

def main(data, model, model_name=None, restore_path=None):

    print("Train images: %d" % (data.train.labels.shape[0]))
    print("Validation images: %d" % (data.validation.labels.shape[0]))
    print("Test images: %d" % (data.test.labels.shape[0]))

    # Create the model
    x = tf.placeholder(tf.float32, [None, 4096, data.train.images.shape[2]])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, data.train.labels.shape[1]])

    # HERE YOU CAN CHANGE THE MODEL TO WHATEVER YA WANT
    y_conv, keep_prob = model(x, data.train.labels.shape[1])

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_,
                                                                logits=y_conv)
    cross_entropy = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        if restore_path:
            # Restore variables from disk.
            saver.restore(sess, "./weights/" + restore_path)
            print("Model restored.")

        merged = tf.summary.merge_all()

        trainPath, testPath = getFileWriterPaths()

        train_writer = tf.summary.FileWriter(trainPath, sess.graph)
        test_writer = tf.summary.FileWriter(testPath)


        for i in range(10):
            batch = data.train.next_batch(32)
            if i % 5 == 0:
                summary, acc = sess.run([merged, accuracy], feed_dict={
                    x: batch[0], y_: batch[1], keep_prob: 1.0})
                test_writer.add_summary(summary, i)
                print('Training accuracy at batch %s: %s' % (i, acc))
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
            train_writer.add_summary(summary, i)

        print('Training done, starts testing')

        run_test_data(data.test.images[:50], data.test.labels[:50], sess, x,
                        keep_prob, y_conv, y_, accuracy, DEBUG = True, WEBCAM_TEST = False)

        if model_name:
            # Save the variables to disk.
            model_name = saver.save(sess, "./weights/" + model_name + ".ckpt")
            print("Model saved in file: %s" % model_name)



def getFileWriterPaths():
    run_id = datetime.datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
    trainPath = os.path.join(env.TENSORBOARD_LOGS_PATH, run_id, "train")
    testPath = os.path.join(env.TENSORBOARD_LOGS_PATH, run_id, "test")

    # Check if using windows
    if (os.name == "nt"):
        # Create folders for windows. Tensorflow already does this for UNIX-based systems
        os.makedirs(os.path.join("logs", run_id, "train"))
        sleep(0.2)
        os.makedirs(os.path.join("logs", run_id, "test"))

    return trainPath, testPath

def directionGestureModelData():
    return direction_gesture_data.read_data_sets(total_samples=520), simple_conv.twolayers

def irModelData():
    return IR_data.read_data_sets(total_samples=520), simple_conv_IR.twolayers

def helpGestureModelData():
    # Note that this implementation does not use RNN.
    # Only same binary classification as for IR data (human vs nohuman)
    return help_gesture_data.read_data_sets(total_samples=520), simple_conv.twolayers

if __name__ == '__main__':
    dictionary = {
        "directionModel": directionGestureModelData,
        "irModel": irModelData,
        "helpModel": helpGestureModelData
    }

    parser=argparse.ArgumentParser()

    parser.add_argument('--model_data', help='Specifiy model and data.')
    #parser.add_argument('--model_name', help='Path to save weights after training.')
    parser.add_argument('--restore_path', help='Path to restore weights before training.')

    args=parser.parse_args()

    # sys.argv contains command line arguments
    data, model = dictionary[args.model_data]();

    main(data, model, model_name=args.model_data, restore_path=args.restore_path)
