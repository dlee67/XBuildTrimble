import urllib.request
import numpy as np
import argparse as ap
import time
import cv2 as cv
import os
import sys
from pathlib import Path
'''
Since, I will be using feature extraction to figure out the displacements,
there is no need to use ML techniques.

Copy pasted from https://youtu.be/z_6fPS5tDNU?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
def store_raw_images():
    tree_images_links = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n13104059"
    tree_images_urls = urllib.request.urlopen(tree_images_links).read().decode()
    print("Looking for TrainingImages")
    if not os.path.exists("TrainingImages"):
        print("TrainingImages not found, creating one.")
        os.makedirs("TrainingImages")
    pic_num = 1
    for i in tree_images_urls.split('\n'):
        print("Beginning parsing.")
        try:
            print("Consuming url: " + i)
            print("pic_num currently is: " + str(pic_num))
            urllib.request.urlretrieve(i, "./TrainingImages/"+str(pic_num)+".jpg")
            img = cv.imread("./TrainingImages/"+str(pic_num)+".jpg", cv.IMREAD_GRAYSCALE)
            resized_image = cv.resize(img, (100, 100))
            cv.imwrite("./TrainingImages/"+str(pic_num)+".jpg", resized_image)
            pic_num = pic_num + 1
            if pic_num == 50:
                break;
        except Exception as e:
            print(str(e))
'''

#Returns the list of file names from the left folder.
def set_target_path():
    listOfFilesLeft = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/left"))
    listOfFilesLeft.sort()
    listOfFilesRight = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/right"))
    listOfFilesRight.sort()
    print("List of extracted files from folder is: " + str(listOfFilesLeft))
    return listOfFilesLeft

#https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
#Extracts features of an image via using KAZE.
def extract_features(image_path, vector_size=32):
    image = cv.imread(image_path, 0)
    alg = cv.KAZE_create()
    kps = alg.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = alg.compute(image, kps)
    dsc = dsc.flatten()
    needed_size = (vector_size * 64)
    if dsc.size < needed_size:
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    return dsc

#From the flattened list, add the values up, so that they are
#represented as 2 dimensional array.
def compacted(flattenedList):
    x = 0
    y = 0
    for index in range(0, flattenedList.size):
        if index < (flattenedList.size/2):
            x = x + flattenedList[index]
        else:
            y = y + flattenedList[index]
    print("X value now: " + str(x))
    print("Y value now: " + str(y))
    return [x, y]

#Prepares the list of 2 dimensional arrays, so that they can be consumed by the matplotlib objects
#for the plotting.
def prepare_data_for_plotting():
    leftFileNames = set_target_path()
    forPlotting = []
    for index in range(0, len(leftFileNames)):
        print("Cosuming file: " + str(os.path.join(os.path.dirname(__file__) + "./images/left/" + leftFileNames[index])))
        extractedValue = extract_features(os.path.join(os.path.dirname(__file__) + "./images/left/" + leftFileNames[index]))
        forPlotting.append(compacted(extractedValue)) # Wow, this actually reads like an English.
    return forPlotting

prepare_data_for_plotting()
#Verifies that images does show.
#someImage = cv.imread(os.path.join(os.path.dirname(__file__) + "./images/left/" + listOfFilesLeft[0]), 0)
#cv.imshow("Die! Kaiser!", someImage)
#cv.waitKey(0);
#cv.destroyAllWindows()
