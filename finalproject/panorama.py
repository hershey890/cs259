import sys
import os
import numpy as np
import cv2

PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    img1 = cv2.imread(PATH + '/data/medium02.jpg')
    img2 = cv2.imread(PATH + '/data/medium03.jpg')

    # stich thses two images
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray, None)
    print(kp1.shape, des1.shape)



if __name__ == '__main__':
    main()