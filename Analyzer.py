import urllib.request
import numpy as np
import argparse as ap
import time
import cv2 as cv
import os
import sys
from pathlib import Path

# Copy pasted from https://youtu.be/z_6fPS5tDNU?list=PLQVvvaa0QuDdttJXlLtAJxJetJcqmqlQq
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

#Prepares a folder with tree images, and makes the tager images recognizable.
def set_target_path():
    listOfFilesLeft = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/left"))
    listOfFilesLeft.sort()
    listOfFilesRight = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/right"))
    listOfFilesRight.sort()
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
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    return dsc

def convert_to_two_d(flattenedList):
    x = 0
    y = 0
    for index in range(0, flattenedList.size):
        if index < (flattenedList.size/2):
            x = x + flattenedList[index]
        else:
            y = y + flattenedList[index]
    print("X value now: " + str(x))
    print("Y value now: " + str(y))

sampleNames = set_target_path()
extracted = extract_features(os.path.join(os.path.dirname(__file__) + "./images/left/" + sampleNames[0]))
convert_to_two_d(extracted)

#Verifies that images does show.
#
#someImage = cv.imread(os.path.join(os.path.dirname(__file__) + "./images/left/" + listOfFilesLeft[0]), 0)
#cv.imshow("Die! Kaiser!", someImage)
#cv.waitKey(0);
#cv.destroyAllWindows()
