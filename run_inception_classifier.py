import tensorflow as tf
import glob
import sys
sys.path.insert(0, './models')
sys.path.insert(0, './inception')

import os
import argparse
import cv2
import numpy as np

import simple_conv
import simple_conv_IR

import classifyWebcamInception

import time

def run_classification(model_name, path_output, path_images, labels, sess, input_layer_name, softmax_tensor, num_top_predictions):

    # Run prediction and draw it in image
    for filename in os.listdir(path_images + "tmp/"):
        #run prediciton on inception net
        image_data = classifyWebcamInception.load_image(path_images + "tmp/" + filename)

        # ts1 = int(round(time.time() * 1000))
        prediction = classifyWebcamInception.run_graph(image_data, labels, input_layer_name, sess, softmax_tensor,
                   num_top_predictions)

        # print("\n")
        # print("classification time (ms):")
        # print((int(round(time.time() * 1000)) - ts1))
        # print("\n")

        img = cv2.imread(path_images + "tmp/" + filename, -1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        if model_name == "directionModel":
            height, width = img.shape[:2]
            middle = int(height/2)
            limit = width
            # (x,y)-format
            classes = [(middle,middle), (41,0), (limit,0), (limit, 21), (limit, 40), (limit, limit), (40,limit), (21,limit), (0,limit), (0,40), (0, 21),(0,0), (24,0)]
            cv2.line(img,classes[int(prediction)],(middle,middle),(255,0,0),1)
            cv2.putText(img,str(prediction),(4,14), font, 0.4, (255,255,255), 1, cv2.LINE_AA)

        else:
            if model_name == "irModel":
                color = (0,0,255)
            else:
                color = (255,255,255)
            cv2.putText(img,str(prediction),(4,14), font, 0.4, color, 1, cv2.LINE_AA)

        cv2.imwrite(path_output + filename,img)
        print(prediction)


    #remove tmp folder content
    files = glob.glob(path_images + "tmp/*")
    for f in files:
        os.remove(f)

    print("See classified images in: " + path_output)
#
# Run the classification on the specified data using the specified models
def run_classifier(model_name, path_images, path_output):

    label_path = "./inception/models/%s/output_labels.txt" % model_name
    graph_path = "./inception/models/%s/output_graph.pb" % model_name

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

    # load labels
    labels = classifyWebcamInception.load_labels(label_path)

    # load graph, which is stored in the default session
    classifyWebcamInception.load_graph(graph_path)

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
        run_classification(model_name, path_output, path_images, labels, sess, 'DecodeJpeg/contents:0', softmax_tensor, 1)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('--model_name', help='Specify model (directionModel, irModel or helpGestureModel).')
    parser.add_argument('--path_images', help='Path to folder containing images to be classified.')
    parser.add_argument('--path_output', help='Path to classified output.')

    args=parser.parse_args()

    run_classifier(args.model_name, args.path_images, args.path_output)
