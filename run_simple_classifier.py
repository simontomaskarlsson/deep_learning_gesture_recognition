import tensorflow as tf
import glob
import sys
sys.path.insert(0, './models')

import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw

import simple_conv
import simple_conv_IR

def cropAndResize(image, pixelDim, heightCrop, widthCrop):
    dim = (pixelDim, pixelDim)
    return cv2.resize(image[pixelDim * heightCrop : pixelDim * heightCrop + pixelDim, pixelDim * widthCrop : pixelDim * widthCrop + pixelDim], dim, interpolation = cv2.INTER_AREA)

def getRestorePath(model):
    # Get and output the correct restore_path depending on model
    restore_path = model + ".ckpt"
    #restore_path = "test.ckpt" #TODO remove
    return restore_path

def getImageArray(img_path):

    file_list = [os.path.basename(x) for x in glob.glob(os.path.join(img_path, "*"))]

    image_array = {}
    counter = 0
    for i in range(0, len(file_list)):
        # Open file
        filename = file_list[i]

        img = Image.open(img_path + filename)#.convert("LA")

        # Put pixel values into an array
        pixels = np.array(img, dtype = np.float32)
        pixels = pixels.reshape(-1, 3)

        if counter == 0:
            image_array = [pixels]
        else:
            image_array = np.append(image_array, [pixels], axis=0)
        counter += 1

    return np.asarray(image_array)

def run_classification(x, images, session, keep_prob, y_conv, model_name, path_output):

    # Run prediction
    predictionsNumpy = session.run(y_conv, feed_dict={x: images, keep_prob: 1})

    # Select the correct class index from each prediction-array
    predictions = [np.argmax(x) for x in predictionsNumpy]

    print(predictions[0:images.shape[0]])

    # Draw image section
    for myIndex in range(len(images)):
        imageArray = np.copy(images[myIndex])

        # image size limit
        limit = int((len(imageArray))**(1/2))

        # Change image format from tensorflow preferences to PIL preferences
        imageArrayFormatted = np.empty((limit,limit,3), dtype=np.uint8)
        for y in range(0, limit):
            for x in range(0, limit):
                imageArrayFormatted[y][x] = [int(pixel) for pixel in imageArray[y*limit + x]]
        im = Image.fromarray(imageArrayFormatted, "RGB")

        middle = limit/2
        draw = ImageDraw.Draw(im)

        if model_name == "directionModel":
            # (x,y)-format
            classes = [(middle,middle), (41,0), (limit,0), (limit, 21), (limit, 40), (limit, limit), (40,limit), (21,limit), (0,limit), (0,40), (0, 21),(0,0), (24,0)]
            draw.line(((middle, middle), classes[predictions[myIndex]]), fill=(0, 0, 250))
            draw.text((1,0), str(predictions[myIndex]), fill=(255,255,255,255))
        else:
            #Draw true or False
            draw.text((1,0), str(predictions[myIndex]), fill=(255,255,255,255))


        im.save(path_output + str(myIndex) + ".png")

# Run the classification on the specified data using the specified models
def run_classifier(model_name, path_images, path_output):

    dictionary = {
        "directionModel": [directionGestureModel, 13],
        "irModel": [irModel, 2],
        "helpModel": [helpGestureModel, 2]
    }
    # sys.argv contains command line arguments
    info = dictionary[model_name];
    model = info[0]()
    nrOfClasses = info[1]

    #Get the model weights
    restore_path = getRestorePath(model_name)

    if not path_output:
        if not os.path.exists(path_images + "classification"):
            os.makedirs(path_images + "classification")
        path_output = path_images + "classification/";

    #Loop over all images in path_images
    for filename in os.listdir(path_images):

        #create tmp folder i path_images
        if not os.path.exists(path_images + "tmp"):
            os.makedirs(path_images + "tmp")

        img = cv2.imread(path_images + filename)

        if img is not None:
            # Crop
            height, width, channels = img.shape
            if width > height:
                diff = width - height
                img = img[0:height,round(diff/2):width-round(diff/2)]
            elif width < height:
                diff = height - width
                img = img[round(diff/2):height-round(diff/2), 0:width]

            #resize
            dim = (64, 64)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

            #store in tmp folder
            cv2.imwrite(path_images + "tmp/" + "preprocessed_" + filename, img)

    #Store all preprocessed images in numpy array
    images = getImageArray(path_images + "tmp/")

    #remove tmp folder
    files = glob.glob(path_images + "tmp/*")
    for f in files:
        os.remove(f)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 4096, images.shape[2]])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, nrOfClasses])

    # HERE YOU CAN CHANGE THE MODEL TO WHATEVER YA WANT
    y_conv, keep_prob = model(x, nrOfClasses)

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
        # Restore variables from disk.
        saver.restore(sess, "./weights/" + restore_path)
        print("Model restored.")

        run_classification(x, images, sess, keep_prob, y_conv, model_name, path_output)

def directionGestureModel():
    return simple_conv.twolayers

def irModel():
    return simple_conv_IR.twolayers

def helpGestureModel():
    # Note that this implementation does not use RNN.
    # Only same binary classification as for IR data (human vs nohuman)
    return simple_conv.twolayers

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('--model_name', help='Specify model (directionModel, irModel or helpGestureModel).')
    parser.add_argument('--path_images', help='Path to folder containing images to be classified.')
    parser.add_argument('--path_output', help='Path to classified output (optional).')

    args=parser.parse_args()

    run_classifier(args.model_name, args.path_images, args.path_output)
