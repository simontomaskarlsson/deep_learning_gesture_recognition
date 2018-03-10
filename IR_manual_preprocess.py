import cv2
import time
import math
import os
import glob
import numpy as np
from PIL import Image
from msvcrt import getch

IMG_PATH = r'D:/MultiRobotsPer/test/'
#
# Change these based on the video source
#
OUTPUT_PATH = 'D:/MultiRobotsPer/test/testlabeled/'
IMAGE_NAME = 'ALhumanVSnonhuman'

#
# Constants used for labeling
#
pixelDim = 64
imageHeight = 332
imageWidth = 440

def cropAndResize(image, pixelDim, heightCrop, widthCrop):
    dim = (pixelDim, pixelDim)
    return cv2.resize(image[pixelDim * heightCrop : pixelDim * heightCrop + pixelDim, pixelDim * widthCrop : pixelDim * widthCrop + pixelDim], dim, interpolation = cv2.INTER_AREA)

def outputFiles(pixelDim, imageHeight, imageWidth):

    for heightCrop in range(0, math.floor(imageHeight/pixelDim)):

        for widthCrop in range(0, math.floor(imageWidth/pixelDim)):
            os.chdir(IMG_PATH)

            for index, oldfile in enumerate(glob.glob("*.png"), start=1):
                os.chdir(IMG_PATH)
                imagefile = cv2.imread(oldfile)
                resized = cropAndResize(imagefile, pixelDim, heightCrop, widthCrop)
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.imshow('image',resized)
                k = cv2.waitKey()

                if k == 49:         # wait for ESC key to exit
                    cv2.destroyAllWindows()
                    newfile = str(heightCrop) + '_'+ str(widthCrop) + '_{}_humanvsnonhuman_1.jpg'.format(index)

                elif k == 48: # wait for 's' key to save and exit
                    cv2.destroyAllWindows()
                    newfile = str(heightCrop) + '_'+ str(widthCrop) + '_{}_humanvsnonhuman_0.jpg'.format(index)

                cv2.imwrite(OUTPUT_PATH + newfile , resized)

if __name__ == '__main__':
    outputFiles(pixelDim, imageHeight, imageWidth)
