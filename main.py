import cv2
import os,sys
import numpy as np
from multiprocessing import Pool
import pytesseract
from pytesseract import Output
import logging
import re

# import ocr.pdf2Image as P2i
# import ocr.document as Doc
# import ocr.imageProcessing as Imp

import pdf2Image as P2i
import document as Doc
import imageProcessing as Imp

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ROOT = os.path.abspath(os.getcwd())
PATH_REPORT = os.path.join(ROOT, 'document-report')

def sortTextOCR(externalBox, widthImage):
    sortCnt = sorted(
        externalBox, key=lambda ctr: ctr[0] + ctr[1] * widthImage)
    return sortCnt

def listDirectory(path):
    listDir = False
    try:
        listDir = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]
    except OSError:
        print("Error: path Error maybe wrong name!!!!")
    return listDir

def tesseractOcr(picture):
    custom_config = r'--oem 1'
    config = custom_config
    text = pytesseract.image_to_string(
        picture, lang='tha+eng', config=custom_config)
    # for add context docx to check error
    cleanText = Doc.cleanTextRegex(text)
    if cleanText:
        return cleanText
    return False

def pipelineOCR(image, page, fileName):
    imageText = Imp.removeBG(image)
    cnts = Imp.findContours(imageText)
    boundaryBox = []
    for cnt in cnts:
        boundaryBox.append(cv2.boundingRect(cnt))
    sortCnts = sortTextOCR(boundaryBox, image.shape[1])
    imageText = cv2.bitwise_not(imageText)
    # imageRepair = repairImage(imageText)
    fulltext = ''
    for inx, box in enumerate(sortCnts):
        x,y,w,h = box
        # repairOCR = imageRepair[y:y+h, x:x+w].copy()
        cropToOCR = imageText[y:y+h, x:x+w].copy()
        text = tesseractOcr(cropToOCR)
        if text:
            fulltext = fulltext + " " + text
    print('finish: ', page)
    Doc.addReportText(fulltext, page, fileName)

def main():
    startPage = 1
    filename = "kati55test"
    name= "./documents-pdf/kati55.pdf"
    page = int(startPage)
    path = P2i.convertPdftoJpg(name,filename,page)
    Doc.createDirectory(PATH_REPORT)
    Doc.createDirectory(PATH_REPORT+"/"+filename)
    poolOCR = Pool(processes=4)
    listPathImage = listDirectory(path)
    for pathImage in listPathImage:
        pageNumber = re.search('page(.*).jpg', pathImage).group(1)
        image = cv2.imread(pathImage)
        poolOCR.apply_async(pipelineOCR, args=(image, pageNumber, filename, ))
    poolOCR.close()
    poolOCR.join()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted ..')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
