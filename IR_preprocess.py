import cv2
import time
import math

OUTPUT_PATH = 'D:/MultiRobotsPer/youtube/2017-12-06/YoutubeCutDone/Images_1/'

#
# Change these based on the video source
#
VIDEO_PATH = 'D:/MultiRobotsPer/youtube/2017-12-06/YoutubeCutDone/2017-12-06-IR-Material_1.mp4'
VIDEO_NAME = 'IR-Material'

FIRST_IMAGE = 0 # Start from this frame index
LAST_IMAGE = float("Inf") # End at this frame index
#
# Constants used for labeling
#
NUMBER_OF_CLASSES = 0
HELP = True # Set true if labeling class 1


def cropAndResize(image):
    dim = (64, 64)
    return cv2.resize(image[0:480, 187:667], dim, interpolation = cv2.INTER_AREA)

    vc = cv2.VideoCapture(VIDEO_PATH)

    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False

        i = 0
        while rval:
            rval, frame = vc.read()

            if i == LAST_IMAGE or not rval:
                print('Crop, resize and labeling done :)')
                break
                i += 1

                if i % 500 == 0:
                    print('Current frame: %d' % (i))

                    if i < FIRST_IMAGE:
                        continue

                        if i == FIRST_IMAGE:
                            print('Starting at frame nr: %d' % (i))
                            #print('Total number of frames: %d' % (int(vc.CAP_PROP_FRAME_COUNT)))

                            if HELP:
                                label_class = '1'
                            else:
                                label_class = '0'

                                resized = cropAndResize(frame)
                                cv2.imwrite(OUTPUT_PATH + str(i) + '_' + VIDEO_NAME + '_' + label_class + '.jpg', resized)
                                cv2.waitKey(1)

                                vc.release()
