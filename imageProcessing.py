import numpy as np
import math
import cv2
from matplotlib import pyplot as plt

def histogram(gray):
    plt.figure()
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.plot(hist, color='k')
    plt.xlim([0, 256])

def countHistogram(gray):
    (unique, counts) = np.unique(gray, return_counts=True)
    print(np.asarray((unique, counts)))

def removeBG(picture):
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    # divide gray by morphology image
    division = cv2.divide(gray, morph, scale=255)
    resize('division', division)
    resize('morph', morph)
    resize('gray', gray)
    histogram(gray)
    histogram(morph)
    histogram(division)
    plt.show()
    # threshold
    return cv2.threshold(division, 0, 255, cv2.THRESH_OTSU +  cv2.THRESH_BINARY_INV )[1]

def repairImage(picture):
    kernalRepair = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    return cv2.morphologyEx(picture, cv2.MORPH_ERODE, kernalRepair, iterations=1)

def dilate(picture,filterShape, w, h, operation=cv2.MORPH_CLOSE, iteration=3):
    kernalDilate = cv2.getStructuringElement(filterShape,(w,h))
    return cv2.morphologyEx(picture, operation, kernalDilate, iterations=iteration)
    
def findContours(picture):
    dilateImage = dilate(picture, cv2.MORPH_ELLIPSE, 13, 5)
    cnts = cv2.findContours(dilateImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def resize(name,picture):
    #resize 
    scale_percent = 40
    width = int(picture.shape[1] * scale_percent / 100)
    height = int(picture.shape[0] * scale_percent / 100)
    #dsize
    dsize = (width, height)
    if width != 0 and height != 0: 
        pictureResize = cv2.resize(picture, dsize)
        cv2.imshow(name,pictureResize)
        cv2.waitKey() 