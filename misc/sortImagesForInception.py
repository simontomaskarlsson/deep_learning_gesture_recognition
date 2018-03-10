import os
import cv2

print("Choose directory where you want the images to be loaded from (e.g: ../data/test/):")
IMG_PATH_OLD = input()

print("Choose directory where you want the images to be saved (e.g: ../data/test/):")
IMG_PATH_NEW = input()

print("Choose what scene should be used as test images (e.g: ../data/test/):")
test_scene = input()

directory = os.path.dirname(IMG_PATH_NEW)
if not os.path.exists(directory):
    os.makedirs(directory)

if not os.path.exists(directory + "/1"):
    os.makedirs(directory + "/1")
    os.makedirs(directory + "/2")
    os.makedirs(directory + "/3")
    os.makedirs(directory + "/4")
    os.makedirs(directory + "/5")
    os.makedirs(directory + "/6")
    os.makedirs(directory + "/7")
    os.makedirs(directory + "/8")
    os.makedirs(directory + "/9")
    os.makedirs(directory + "/10")
    os.makedirs(directory + "/11")
    os.makedirs(directory + "/12")
    os.makedirs(directory + "/0")
    os.makedirs(directory + "/testimages")

counter = 1
for filename in os.listdir(IMG_PATH_OLD):

    image = cv2.imread(IMG_PATH_OLD + filename)

    info = filename.split(".")[0].split("_")
    label = info[-1]

    if str(info[1])==test_scene:
        cv2.imwrite(IMG_PATH_NEW + "testimages" + "/" + filename, image)
    else:
        cv2.imwrite(IMG_PATH_NEW + label + "/" + filename, image)

    if counter % 100 == 0:
        print(str(counter))
    counter += 1
