import urllib.request
import numpy as np
import argparse as ap
import matplotlib.pyplot as plt
import time
import cv2
import os
import sys
from pathlib import Path

#Returns the list of file names from the left folder.
def return_left_samples():
    listOfFilesLeft = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/left"))
    listOfFilesLeft.sort()
    return listOfFilesLeft

#Returns the list of file names from the right folder.
def return_right_samples():
    listOfFilesRight = os.listdir(os.path.join(os.path.dirname(__file__) + "./images/right"))
    listOfFilesRight.sort()
    return listOfFilesRight

#https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
#Extracts features of an image via using KAZE.
def extract_features(image_path, vector_size=32):
    print("The passed image path is: " + str(image_path))
    image = cv2.imread(image_path, 0)
    alg = cv2.KAZE_create()
    kps = alg.detect(image)
    kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
    kps, dsc = alg.compute(image, kps)
    needed_size = (vector_size * 64)
    return kps

def prep_left_and_right_samples():
    leftImgDir = "./images/left/" + str(return_left_samples()[0])
    rightImgDir = "./images/right/" + str(return_right_samples()[0])
    img1 = cv2.imread(leftImgDir, 0)
    img2 = cv2.imread(rightImgDir, 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
    plt.imshow(img3)
    plt.show()

prep_left_and_right_samples()

#Objects has dots.
#def to_harris():
#    listOfFilesLeft = set_target_path()
#    filename = os.path.join(os.path.join(currentPath + "./images/left/" + listOfFilesLeft[0]))
#    img = cv2.imread(filename)
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    gray = np.float32(gray)
#    dst = cv2.cornerHarris(gray,100,3,0.04)
#    #result is dilated for marking the corners, not important
#    dst = cv2.dilate(dst,None)
#    # Threshold for an optimal value, it may vary depending on the image.
#    img[dst>0.01*dst.max()]=[0,0,255]
#    img = cv2.resize(img, (800, 500))
#    cv2.imshow('dst',img)
#    if cv2.waitKey(0) & 0xff == 27:
#        cv2.destroyAllWindows()
