import os

path = 'C:/Users/Per Welander/Development/safe/2017-11-24/IR/AllPreprocessed'
wantedstr0 = '_0'
wantedstr1 = '_1'

counter0 = 0
counter1 = 0
for file in os.listdir(path):
    if file.endswith(".jpg"):
        if file.find(wantedstr0) > 0:
            counter0 = counter0 + 1
            print(counter0)
        if file.find(wantedstr1) > 0:
            counter1 = counter1 + 1
            print(counter1)

if counter == 0:
    print("No file has been found")
