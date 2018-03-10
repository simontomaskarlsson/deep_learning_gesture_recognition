import os

path = 'data/test_IR_non_human/'
oldstr = 'WalkInPark_1'
newstr = 'WalkInPark_0'

counter = 0
for file in os.listdir(path):
    if file.endswith(".jpg"):
        if file.find(oldstr) > 0:
            counter = counter + 1
            os.rename(path + "\\"+file, path + "\\"+file.replace(oldstr, newstr))

if counter == 0:
    print("No file has been found")
