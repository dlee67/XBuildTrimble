import urllib.request
import numpy as np
import argparse as ap
import time
import cv2
import os
import sys
from pathlib import Path

listOfFilesLeft = []
listOfFilesRight = []
currentPath = os.path.join(os.path.dirname(__file__))

#Returns the list of file names from the left folder.
def set_target_path():
    listOfFilesLeft = os.listdir(os.path.join(currentPath + "./images/left"))
    listOfFilesLeft.sort()
    listOfFilesRight = os.listdir(os.path.join(currentPath + "./images/right"))
    listOfFilesRight.sort()
    return listOfFilesLeft

#Objects has dots.
def to_harris():
    listOfFilesLeft = set_target_path()
    filename = os.path.join(os.path.join(currentPath + "./images/left/" + listOfFilesLeft[0]))
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,100,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[0,0,255]
    img = cv2.resize(img, (800, 500))
    cv2.imshow('dst',img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()


#https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
#Extracts features of an image via using KAZE.
def extract_features(image_path, vector_size=32):
    print("The passed image path is: " + str(image_path))
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

set_target_path()
to_harris()
