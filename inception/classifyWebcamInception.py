from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from PIL import Image, ImageDraw


import glob
import os

import argparse
import sys
import time
import os

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument(
    '--image', type=str, help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    required=True,
    type=str,
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    required=True,
    type=str,
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

def run_graph(image_data, labels, input_layer_name, sess, softmax_tensor,
              num_top_predictions):

    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      #print('%s (score = %.5f)' % (human_string, score))

    return int(human_string)

def startWebcam(classificationFunc):
    waitingLimit = 30
    predictionText = ""

    myIndex = 0


    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        waitingLimit -= 1


        # image size limit
        limitHeight = len(frame)
        limitWidth = len(frame[0])
        im = Image.fromarray(frame, "RGB")
        middleHeight = int(limitHeight/2)
        middleWidth = int(limitWidth/2)

        #draw = ImageDraw.Draw(im)

        rval, frame = vc.read()
        if waitingLimit == 15:
            # Save image
            # files = glob.glob('data/webcam/*')
            # for f in files:
            #     os.remove(f)

            widthCut = int((limitWidth - limitHeight)/2)
            resizedFrame = cv2.resize(frame[0:limitHeight, widthCut:limitWidth-widthCut], (64,64), interpolation = cv2.INTER_AREA)
            resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            flattenedResizedFrame = np.array([[item for sublist in resizedFrame for item in sublist]])
            resizedImage = Image.fromarray(resizedFrame, "RGB")
            resizedImage.save("./data/webcam/temp.jpg")

        if waitingLimit == 0:
            # Predict saved image
            widthCut = int((limitWidth - limitHeight)/2)
            resizedFrame = cv2.resize(frame[0:limitHeight, widthCut:limitWidth-widthCut], (64,64), interpolation = cv2.INTER_AREA)
            resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            #resizedFrame = cv2.resize(frame,(64, 64), interpolation = cv2.INTER_AREA)

            flattenedResizedFrame = np.array([[item for sublist in resizedFrame for item in sublist]])

            # Select the correct class index from each prediction-array
            # predictionsNumpy = session.run(y_conv, feed_dict={x: flattenedResizedFrame, keep_prob: 1})

            loaded_image = load_image("./data/webcam/temp.jpg")
            predictions = classificationFunc(loaded_image)

            # print(predictions)

            predictionText = "class: " + str((int(predictions) + 1) % 2) #flipping class

            #resizedImage = Image.fromarray(resizedFrame, "RGB")
            #resizedImage.save("./data/webcam/image" + str(myIndex) + "_rasmusindoor3_" + "1"+ ".PNG",)



            myIndex += 1 # remove me later

            waitingLimit = 30

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,predictionText,(0,50), font, 1, (0,0,0), 2, cv2.LINE_AA)

        cv2.imshow("preview", frame)
        #cv2.putText(frame,'OpenCV Tuts!',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)
        #cv2.line(frame,(0,0),(middleWidth,middleHeight),(255,255,255),3)
        #cv2.rectangle(frame,(500,250),(1000,500),(0,0,255),15)
        #cv2.circle(frame,(447,63), 63, (0,255,0), -1)

        limitHeight = len(frame)
        limitWidth = len(frame[0])


        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")

def main(argv):
    """Runs inference on an image."""
    if argv[1:]:
        raise ValueError('Unused Command Line Args: %s' % argv[1:])

    #if not tf.gfile.Exists(image):
    #    tf.logging.fatal('image file does not exist %s', FLAGS.image)

    if not tf.gfile.Exists(FLAGS.labels):
        tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

    if not tf.gfile.Exists(FLAGS.graph):
        tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

    # load labels
    labels = load_labels(FLAGS.labels)

    # load graph, which is stored in the default session
    load_graph(FLAGS.graph)

    #image_data = #insert here
    #prediction = run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
    #        FLAGS.num_top_predictions)
    with tf.Session() as sess:
       # Feed the image_data as input to the graph.
       #   predictions will contain a two-dimensional array, where one
       #   dimension represents the input image count, and the other has
       #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(FLAGS.output_layer)
        startWebcam(lambda image: run_graph(image, labels, FLAGS.input_layer, sess, softmax_tensor,
            FLAGS.num_top_predictions))



if __name__ == "__main__":
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=sys.argv[:1]+unparsed)
