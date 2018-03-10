import cv2
import os

print("Choose directory where you want the images to be saved (e.g: ../data/test/):")
IMG_PATH = input()

print("Choose dimension on new images (image will be MxM, enter value on M):")
dim_input = int(input())

print("Do you want to convert images to grayscale (y/n)?")
convert_grayscale = input()

directory = os.path.dirname(IMG_PATH)
if not os.path.exists(directory):
    os.makedirs(directory)

counter = 1
for filename in os.listdir('data/images/'):
    image = cv2.imread("data/images/" + filename)
    if convert_grayscale=="y":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dim = (dim_input,dim_input)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(IMG_PATH + filename, resized)

    if counter % 100 == 0:
        print(str(counter) + " out of ~26k")
    counter += 1
