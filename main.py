import cv2
import os,sys
import numpy as np
import pytesseract
from pytesseract import Output
from matplotlib import pyplot as plt

import document as Doc
import pdf2Image as P2i
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# func rotate
TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour
TEXT_MAX_HEIGHT = 100 
########################

def histogram(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    plt.plot(hist, color='k')
    plt.xlim([0, 256])
    plt.show()

def countHistogram(gray):
    (unique, counts) = np.unique(gray, return_counts=True)
    print(np.asarray((unique, counts)))

def findContours(picture, parameterRETR,parameterChain):
    cnts = cv2.findContours(picture, parameterRETR, parameterChain)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def removeBG(picture):
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT , (5,5))
    morph = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel)
    # divide gray by morphology image
    division = cv2.divide(gray, morph, scale=255)
    # threshold
    return cv2.threshold(division, 0, 255, cv2.THRESH_OTSU +  cv2.THRESH_BINARY_INV )[1]

def removeLine(picture):
    # set kernel for remove horizontal line & vertical line
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(picture, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detected_lines2 = cv2.morphologyEx(picture, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

    #find contours & remove line with white color 
    cntsHorizontal = findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsVertical = findContours(detected_lines2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cntsHorizontal:
        cv2.drawContours(picture, [c], -1, (255,255,255), -1)
    for c in cntsVertical:
        cv2.drawContours(picture, [c], -1, (255,255,255), -1)
    return picture 

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

def sortTextOCR(externalBox, widthImage):
    sortCnt = sorted(
        externalBox, key=lambda ctr: ctr[0] + ctr[1] * widthImage)
    return sortCnt


def tesseractOcr(picture):
    # imageOCR = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 1'
    config = custom_config
    # textconfident = pytesseract.image_to_data(
    # imageOCR, lang = 'tha+eng', output_type = Output.DICT, config = custom_config)
    # textcon = CalculateTextConfident(textconfident)
    text = pytesseract.image_to_string(
        picture, lang='tha+eng', config=custom_config)
    # for add context docx to check error
    cleanText = Doc.cleanTextRegex(text)
    if cleanText:
        return cleanText
    return "NONONO"

def repairImage(picture):
    kernalRepair = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    return cv2.morphologyEx(picture, cv2.MORPH_HITMISS, kernalRepair, iterations=1)

def main():
    # path = P2i.convertPdftoJpg('./documents-pdf/kati56.pdf','kati56')\
    BLUE = [255,255,255]
    image = cv2.imread('./documents-image/kati50/page22.jpg')
    imageText = removeBG(image)    
    kernalDilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,5))
    dilate = cv2.morphologyEx(imageText, cv2.MORPH_CLOSE, kernalDilate, iterations=3)
    cnts = findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boundaryBox = []
    for cnt in cnts:
        boundaryBox.append(cv2.boundingRect(cnt))

    sortCnts = sortTextOCR(boundaryBox, image.shape[1])
    # resize('before', dilate)

    imageText = cv2.bitwise_not(imageText)
    imageRepair = repairImage(imageText)

    arrayPicture = []
    arrayText = []
    arrayText2 = []

    for inx, box in enumerate(sortCnts):
        x,y,w,h = box
        if h < TEXT_MIN_HEIGHT or w < TEXT_MIN_WIDTH: 
            print("paste")
        else:
            repairOCR = imageRepair[y:y+h, x:x+w].copy()
            cropToOCR = imageText[y:y+h, x:x+w].copy()
                
            image = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0),2)      
            # cropToOCR= cv2.copyMakeBorder(cropToOCR.copy(),5,5,5,5,cv2.BORDER_CONSTANT,value=BLUE)
            # repairOCR= cv2.copyMakeBorder(repairOCR.copy(),5,5,5,5,cv2.BORDER_CONSTANT,value=BLUE)

            arrayPicture.append(cropToOCR)
            arrayText.append(tesseractOcr(cropToOCR))
            arrayText2.append(tesseractOcr(repairOCR))
            cv2.destroyAllWindows()
    resize('sd', image)

    Doc.addReportDoc(arrayPicture, arrayText,arrayText2,'./repch.docx')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted ..')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
