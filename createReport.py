import os, sys
import multiprocessing as mp
import pytesseract
from pytesseract import Output
import cv2
import logging
import tensorflow as tf
import numpy as np

import imageProcessing as Imp
import pdf2Image as PI
import document as Doc
import tokenizer
 
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
    docPath = pathFolderImg + '/' + 'page-' + str(page) +'.docx'
    # mydoc = Doc.createDoc(docPath)
    logging.info("FileName: " + str(folderName)+ " Page: "+str(page)+ " Start image process")
    arrayText = []
    arrayImage = []
    arrayCleanText = []
    skipPage = Imp.skipPage(image)
    if skipPage:
        return

    imageOnlyText, angleBox, externalBox = Imp.prepareRotated(image)
    imageRemoveLine = Imp.removeLine(imageOnlyText)
    sortExternalBox = sortTextOCR(externalBox, imageRemoveLine.shape[1])
    ### clean memory
    del imageOnlyText
    del externalBox
    del skipPage
    del image

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices(
                    'GPU')
                print(len(gpus), "Physical GPUs,", len(
                    logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    for inx, box in enumerate(sortExternalBox):
        cleanSentence = []
        imageRotated = Imp.rotated(imageRemoveLine, angleBox, inx, box)
        text = tesseractOcr(imageRotated)
        if text != '<EMP>':
            cleanSentence = tokenizer.cleanWord(text)
            cleanSentence = list(map(tokenizer.spellCheckAuto, cleanSentence))
            tempArray = cleanSentence.copy()
            if len(tempArray) >= 1:
                arrayCleanText.append(tempArray)
            else:
                arrayCleanText.append('<EMP>')
            del tempArray
        else:
            arrayCleanText.append('<EMP>')
        # img = Doc.saveTempImg(imageRotated)
        ### tesseract already send token if empty word
        # arrayImage.append(img)
        arrayImage.append(imageRotated)
        arrayText.append(text)

    ### clean memory
    del sortExternalBox
    del imageRemoveLine

    Doc.addReportDoc(arrayImage, arrayText, arrayCleanText, docPath)
    logging.info("FileName: " + str(folderName)+ " Page: "+str(page)+ " Finish add report")
    return

def main():
    pool = mp.Pool(processes=4)
    createDirectory(PDF_FOLDERPATH)
    createDirectory(IMAGE_FOLDERPATH)
    convertImg = input(f'you want to convert PDF to img (y/n)> ')
    if convertImg == 'y':
        pathImage, imageName = convertPdf()
    else:
        imageName = input(f'Enter your image folder name (only in documents-image!!!!)> ')
        pathImage = IMAGE_FOLDERPATH + imageName
    listPathImage = listDirectory(pathImage)
    if not listPathImage:
        return
    reportPath = './report'
    pathFolderImg = reportPath + '/' + imageName
    createDirectory(reportPath)
    createDirectory(pathFolderImg)
    logging.info("FileName: " + str(imageName) + "Start process")
    for inx, path in enumerate(listPathImage):
        pool.apply_async(pipeLineOcr, args=(path, pathFolderImg, str(inx+1), imageName, ))
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

    