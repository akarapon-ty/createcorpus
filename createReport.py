import os, sys
import multiprocessing as mp
import pytesseract
from pytesseract import Output
import cv2
import logging
# import tensorflow as tf
import numpy as np
import re

import imageProcessing as Imp
import pdf2Image as PI
import document as Doc
import tokenizer
font = cv2.FONT_HERSHEY_SIMPLEX 

    ### Plan
    # 1. select to convert PDF or use pic
    # 2. list directory open mp
    # 3. mp do image process -> ocr -> clean text -> save to word

PDF_FOLDERPATH = './documents-pdf/'
IMAGE_FOLDERPATH = './documents-image/'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ROOT = os.path.abspath(os.getcwd())

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(processName)-8s %(message)s',
                    datefmt='%d-%m-%y %H:%M',
                    filename='./processLog.log',
                    filemode='a')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(processName)-12s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def listDirectory(path):
    listDir = False
    try:
        listDir = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]
    except OSError:
        logging.info("Error: path Error maybe wrong name!!!!")
    return listDir

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

def convertPdf():
    pdfName = input(f'Enter your PDF name (only in documents-pdf!!!!)> ')
    pathPdf = PDF_FOLDERPATH + pdfName + '.pdf'
    pathImage = PI.convertPdftoJpg(pathPdf, pdfName)
    return pathImage, pdfName

def sortTextOCR(externalBox, widthImage):
    sortCnt = sorted(
        externalBox, key=lambda ctr: ctr[0] + ctr[1] * widthImage)
    return sortCnt

def tesseractOcr(picture):
    imageOCR = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    custom_config = r'--oem 1'
    config = custom_config
    text = pytesseract.image_to_string(
        imageOCR, lang='tha+eng', config=custom_config)
    # for add context docx to check error
    cleanText = Doc.cleanTextRegex(text)
    if cleanText:
        return cleanText
    return "<EMP>"

def pipeLineOcr(pathImage, pathFolderImg, page, folderName):
    image = cv2.imread(pathImage)
    logging.info("FileName: " + str(folderName)+ " Page: "+str(page)+ " Start image process")
    arrayText = []
    arrayImage = []
    arrayCleanText = []
    skipPage = Imp.skipPage(image)
    if skipPage:
        cv2.putText(image,"SKIP", (x,y), 0,font, 2, 255)
        cv2.imwrite('./readpic/'+ str(folderName) +'/'+str(page)+'.jpg', image)
        return

    imageOnlyText, angleBox, externalBox = Imp.prepareRotated(image)
    imageRemoveLine = Imp.removeLine(imageOnlyText)
    sortExternalBox = sortTextOCR(externalBox, imageRemoveLine.shape[1])
    ### clean memory
    del imageOnlyText
    del externalBox
    del skipPage

   

    for inx, box in enumerate(sortExternalBox):
        cleanSentence = []
        imageRotated = Imp.rotated(imageRemoveLine, angleBox, inx, box)
        x,y,w,h = box
        image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0),2)
    cv2.imwrite('./readpic/'+ str(folderName) +'/'+str(page)+'.jpg', image)
    ### clean memory
    del sortExternalBox
    del imageRemoveLine

    logging.info("FileName: " + str(folderName)+ " Page: "+str(page)+ " Finish add report")
    return

def main():
    pool = mp.Pool(processes=5)
    convertImg = input(f'you want to convert PDF to img (y/n)> ')
    if convertImg == 'y':
        pathImage, imageName = convertPdf()
    else:
        imageName = input(f'Enter your image folder name (only in documents-image!!!!)> ')
        pathImage = IMAGE_FOLDERPATH + imageName
    listPathImage = listDirectory(pathImage)
    if not listPathImage:
        return
    reportPath = './readpic'
    pathFolderImg = reportPath + '/' + imageName
    createDirectory(reportPath)
    createDirectory(pathFolderImg)
    logging.info("FileName: " + str(imageName) + "Start process")
    for inx, path in enumerate(listPathImage):
        pageNumber = re.search('page(.*).jpg', path).group(1)
        pool.apply(pipeLineOcr, args=(path, pathFolderImg, pageNumber, imageName, ))
    pool.close()
    pool.join()
    logging.info("FileName: " + str(imageName) + "Finish process")


if __name__ == '__main__':
   try:
     main()
   except KeyboardInterrupt:
     print ('\nInterrupted ..')
     try:
       sys.exit(0)
     except SystemExit:
       os._exit(0)

    