import cv2
import time
import math
import os

OUTPUT_PATH_64 = 'data/tmp64/'
OUTPUT_PATH_256 = 'data/tmp256/'

directory64 = os.path.dirname(OUTPUT_PATH_64)
if not os.path.exists(directory64):
    os.makedirs(directory64)

directory256 = os.path.dirname(OUTPUT_PATH_256)
if not os.path.exists(directory256):
    os.makedirs(directory256)

#
# Change these based on the video source
#
VIDEO_PATH = 'data/videos/asfaltWithLines.MP4'
VIDEO_NAME = 'asfaltWithLines'
 
twelves = [154, 243, 330, 417, 504, 600, 693, 796, 897, 986, 1088]

for i, _ in enumerate(twelves):

    if i == len(twelves)-1:
        break;

    IMG_PER_SPIN = -(twelves[i+1]-twelves[i]) # Number of images per spin
    SPIN_REFERENCE = twelves[i] # 154 243 330 417 504 600 693 796 897 986 1088  # Index of image with hand pointed at 0 degrees


    FIRST_IMAGE = twelves[i] # Start from this frame index
    LAST_IMAGE = twelves[i+1]-1 # End at this frame index

    #
    # Constants used for labeling
    #
    NUMBER_OF_CLASSES = 12
    NO_POINTING = False # Set true if labeling class 0

    def label(spin_ref, img_per_spin, number_of_classes, currentIndex):
        img_per_class = img_per_spin/number_of_classes
        label_class = math.ceil(((currentIndex - spin_ref)/img_per_class) % number_of_classes)

        if label_class == 0:
            label_class = 1

        return str(label_class)

    def cropAndResize64(image):
        dim = (64, 64)
        return cv2.resize(image[0:720, 280:1000], dim, interpolation = cv2.INTER_AREA)

    def cropAndResize256(image):
        dim = (256, 256)
        return cv2.resize(image[0:720, 280:1000], dim, interpolation = cv2.INTER_AREA)


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

        if NO_POINTING:
            label_class = '0'
        else:
            label_class = label(SPIN_REFERENCE, IMG_PER_SPIN, NUMBER_OF_CLASSES, i)

        resized64 = cropAndResize64(frame)
        resized256 = cropAndResize256(frame)

        cv2.imwrite(OUTPUT_PATH_64 + str(i) + '_' + VIDEO_NAME + '_' + label_class + '.jpg', resized64)
        cv2.imwrite(OUTPUT_PATH_256 + str(i) + '_' + VIDEO_NAME + '_' + label_class + '.jpg', resized256)
        cv2.waitKey(1)


    vc.release()
