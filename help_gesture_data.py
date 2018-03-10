import numpy as np
import collections
import os
import glob
import cv2
import random
from PIL import Image


class DataSet(object):
    def __init__(self, images, labels):
        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def read_data_sets(total_samples=None, train_file="./help_gesture_train_data_temp.npz",
                    test_file = "./help_gesture_test_data_temp.npz"):
    # Number of training and evaluation data
    IMG_PATH = "./data/images/helpGesture/"

    # Attempt to read .npz-files, if they exist
    if os.path.isfile(train_file) and os.path.isfile(test_file):
        print("Loading from file %s" % (train_file))
        train_saved = np.load(train_file)
        print("Loading from file %s" % (test_file))
        test_saved = np.load(test_file)
        return split_into_datasets(train_saved["images"], train_saved["labels"],
                                   test_saved["images"], test_saved["labels"])

    # Split files into train and test
    all_files_list = [os.path.basename(x) for x in glob.glob(os.path.join(IMG_PATH, "*"))]
    train_files = [x for x in all_files_list if "simonWebcam2" not in x]
    test_files = [x for x in all_files_list if "simonWebcam2" in x]

    if not train_files:
        raise Exception('No train images found.')

    if not test_files:
        raise Exception('No test images found.')

    train_images, train_labels = files_to_images(train_files, total_samples, IMG_PATH, train_file)
    test_images, test_labels = files_to_images(test_files, total_samples, IMG_PATH, test_file)

    return split_into_datasets(train_images, train_labels, test_images, test_labels)

def files_to_images(file_list, total_samples, IMG_PATH, TEMP_FILE_NAME):

    # Get distrubution of image labels
    images_by_label = {}
    dist_by_label = {}
    for i in range(len(file_list)):
        label = get_image_info(file_list[i])["label"]
        if label not in images_by_label:
            images_by_label[label] = [i]
            dist_by_label[label] = 1
        else:
            images_by_label[label].append(i)
            dist_by_label[label] += 1

    # Get label with least amount of images
    min_labels = dist_by_label[min(dist_by_label, key=dist_by_label.get)]

    if total_samples:
        min_labels_total_samples = int(total_samples/len(dist_by_label.keys()))
        if min_labels_total_samples < min_labels: # not enought samples to fill out total_samples
            min_labels = min_labels_total_samples

    weighed_seq = []
    for label, images in images_by_label.items():
        print(len(images))
        weighed_seq += random.sample(images, min_labels)
    np.random.shuffle(weighed_seq)

    weighed_total_images = len(weighed_seq)
    counter = 0
    print("Starting to process %d images" % (weighed_total_images))
    for i in weighed_seq:
        # Open file and convert it to greyscale
        filename = file_list[i]

        img = Image.open(IMG_PATH + filename)#.convert("LA")
        image_info = get_image_info(filename)
        label = image_info["label"]

        label_probs = np.zeros(shape=(len(dist_by_label.keys()),))
        label_probs[label] = np.float32(1.0)

        # Put pixel values into an array
        pixels = np.array(img, dtype = np.float32)
        pixels = pixels.reshape(-1, 3)
        #pixels = np.delete(pixels, np.s_[1:3], 1)

        if counter == 0:
            images = [pixels]
            labels = [label_probs]
        else:
            images = np.append(images, [pixels], axis=0)
            labels = np.append(labels, [label_probs], axis=0)
        if counter % 100 == 0:
            print("Image nr %d processed" % (counter))
        counter += 1

    # Normalize images
    #images = normalize(images)

    np.savez(TEMP_FILE_NAME, images=images, labels=labels)

    return images, labels


def get_image_info(image_name):
    info = image_name.split(".")[0].split("_")
    return {"id": info[0], "label": int(info[-1])}

def normalize(x):
    maximum = np.max(x)
    minimum = np.min(x)
    return (x - minimum) / (maximum - minimum)

def split_into_datasets(train_images, train_labels, test_images, test_labels):
    #validation_split = int(images.shape[0]*0.5)
    #test_split = int(images.shape[0]*0.75)

    #train_images = images[:validation_split]
    train_mean = np.mean(train_images, axis = 0) # Mean subtraction on train images
    train_images -= train_mean
    #train_labels = labels[:validation_split]

    #validation_images = images[validation_split:test_split]
    #validation_images -= train_mean
    #validation_labels = labels[validation_split:test_split]

    #test_images = images[test_split:]
    test_images -= train_mean
    #test_labels = labels[test_split:]

    train = DataSet(train_images, train_labels)
    validation = train #DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    Datasets = collections.namedtuple("Datasets", ["train", "validation", "test"])
    return Datasets(train=train, validation=validation, test=test)

# def split_into_datasets(images, labels):
#     validation_split = int(images.shape[0]*0.5)
#     test_split = int(images.shape[0]*0.75)
#
#     # Mean subtraction on train images
#     #images -= np.mean(images, axis = 0)
#     train_images = images[:validation_split]
#     train_labels = labels[:validation_split]
#     validation_images = images[validation_split:test_split]
#     validation_labels = labels[validation_split:test_split]
#     test_images = images[test_split:]
#     test_labels = labels[test_split:]
#
#     train = DataSet(train_images, train_labels)
#     validation = DataSet(validation_images, validation_labels)
#     test = DataSet(test_images, test_labels)
#
#     Datasets = collections.namedtuple("Datasets", ["train", "validation", "test"])
#     return Datasets(train=train, validation=validation, test=test)

if __name__ == "__main__":
    data = read_data_sets()
    print(len(data.train.images[1]))
    print((data.train.images[1]).shape)
