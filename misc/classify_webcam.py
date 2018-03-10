import cv2
import numpy as np
from PIL import Image, ImageDraw


import glob
import os

def startWebcam(session, x, y_conv, keep_prob):
    waitingLimit = 30
    predictionText = ""

    #files = glob.glob('data/webcam/*')
    #for f in files:
    #    os.remove(f)

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

        if waitingLimit == 0:
            widthCut = int((limitWidth - limitHeight)/2)
            resizedFrame = cv2.resize(frame[0:limitHeight, widthCut:limitWidth-widthCut], (64,64), interpolation = cv2.INTER_AREA)
            resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
            #resizedFrame = cv2.resize(frame,(64, 64), interpolation = cv2.INTER_AREA)

            flattenedResizedFrame = np.array([[item for sublist in resizedFrame for item in sublist]])

            # Select the correct class index from each prediction-array
            predictionsNumpy = session.run(y_conv, feed_dict={x: flattenedResizedFrame, keep_prob: 1})
            predictions = np.argmax(predictionsNumpy[0])

            # print(predictions)

            predictionText = "class: " + str(predictions)

            resizedImage = Image.fromarray(resizedFrame, "RGB")
            resizedImage.save("./data/webcam/image" + str(myIndex) + "_rasmusindoor3_" + "1"+ ".PNG",)
            myIndex += 1 # remove me later

            waitingLimit = 2

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

if __name__ == '__main__':
    startWebcam()
