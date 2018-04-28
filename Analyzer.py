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

#Returns list of files names in the left and right folder;
#however, only one is required.
listOfFilesLeft = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/left"))
listOfFilesLeft.sort()
listOfFilesRight = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/right"))
listOfFilesRight.sort()

store_raw_images()

#for index in listOfFilesLeft:
#    someImage = cv.imread(os.path.join(os.path.dirname(__file__) + "./images/left/" + listOfFilesLeft[0]), 0)

'''
Verifies that images does show.

someImage = cv.imread(os.path.join(os.path.dirname(__file__) + "./images/left/" + listOfFilesLeft[0]), 0)
cv.imshow("Die! Kaiser!", someImage)
cv.waitKey(0);
cv.destroyAllWindows()
'''
