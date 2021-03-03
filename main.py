import cv2
import os,sys
import numpy as np
from multiprocessing import Pool, Manager
import pytesseract
import threading
from pytesseract import Output
import logging
import re
import tensorflow as tf

# import ocr.pdf2Image as P2i
# import ocr.document as Doc
# import ocr.imageProcessing as Imp

import pdf2Image as P2i
import document as Doc
import imageProcessing as Imp
import tokenizer as token

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
ROOT = os.path.abspath(os.getcwd())
PATH_REPORT = os.path.join(ROOT, 'document-report')

# func rotate
TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour
TEXT_MAX_HEIGHT = 100 
########################

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

def sortTextOCR(externalBox, widthImage):
    sortCnt = sorted(
        externalBox, key=lambda ctr: ctr[0] + ctr[1] * widthImage)
    return sortCnt


def listDirectory(path):
    listDir = False
    try:
        onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        # listDir = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".jpg")]
    except OSError:
        logging.info("Error: path Error maybe wrong name!!!!")
    return onlyfiles

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

def pipeLineConsumer(queueWriteFile, queueWork, path):
    while True:
        document = queueWriteFile.get()
        queueWork.put(1)
        Doc.writeFile(path, document)
        # logging.debug("Page:" + str(re.search('page-(.*).txt', path).group(1)) + " Finish write File")
        if queueWriteFile.empty():
            queueWork.get()

def pipelineOCR(image, page, fileName, queueWriteFile):
    ### set limit growth memory gpu
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

    fileName = re.search('(.*).pdf', fileName).group(1)
    imageText = Imp.removeBG(image)
    cnts = Imp.findContours(imageText)
    boundaryBox = []
    for cnt in cnts:
        boundaryBox.append(cv2.boundingRect(cnt))
    sortCnts = sortTextOCR(boundaryBox, image.shape[1])
    imageText = cv2.bitwise_not(imageText)
    # imageRepair = repairImage(imageText)
    fullTextClean = []
    pathSaveClean = PATH_REPORT+"/"+fileName+"/"+"page-"+str(page)+".txt"
    Doc.createFileCorpus(pathSaveClean)
    logging.info("FileName: " + str(fileName)+ " Page: "+str(page)+ " Start OCR")
    for inx, box in enumerate(sortCnts):
        x,y,w,h = box
        if h < TEXT_MIN_HEIGHT or w < TEXT_MIN_WIDTH:
            pass
        # repairOCR = imageRepair[y:y+h, x:x+w].copy()
        cropToOCR = imageText[y:y+h, x:x+w].copy()
        text = tesseractOcr(cropToOCR)
        if text:
            cleanSentence = token.cleanWord(text)
            cleanSentence = list(map(token.spellCheckAuto, cleanSentence))
            fullTextClean.extend(cleanSentence)
            if len(cleanSentence) > 1:
                cleanSentence.append('\n')
                Doc.writeFile(pathSaveClean, cleanSentence)
    logging.info("FileName: " + str(fileName)+ " Page: "+str(page)+ " Finish OCR")
    queueWriteFile.put(fullTextClean)

def main():
    startPage = 1
    # filename = "kati55test"
    # name= "./documents-pdf/kati55.pdf"
    
    #### create consumer thread to write corpus file
    pathCorpusFile="./corpus.txt"
    manager = Manager()
    queueWriteFile = manager.Queue()
    queueWork = manager.Queue(1)
    consumer = threading.Thread(target=pipeLineConsumer, args=(queueWriteFile, queueWork, pathCorpusFile, ))
    consumer.setDaemon=True
    consumer.start()
    #######

    Doc.createFileCorpus(pathCorpusFile)
    pathPDF = "D:\word2vec\createcorpus\documents-pdf"
    listFilePDF = listDirectory(pathPDF)
    for pdfName in listFilePDF:
        page = int(startPage)
        filename = pdfName
        pdf = pathPDF + "/" + pdfName
        path = P2i.convertPdftoJpg(pdf,filename,page)
        Doc.createDirectory(PATH_REPORT)
        poolOCR = Pool(processes=4)
        listPathImage = listDirectory(path)
        Doc.createDirectory(PATH_REPORT+"/"+re.search('(.*).pdf', filename).group(1))
        for ImageName in listPathImage:
            pathImage = path + '/' +ImageName
            pageNumber = re.search('page(.*).jpg', pathImage).group(1)
            image = cv2.imread(pathImage)
            poolOCR.apply_async(pipelineOCR, args=(image, pageNumber, filename, queueWriteFile))
        poolOCR.close()
        poolOCR.join()
        while True:
            if queueWork.empty():
                logging.info('FolderName: ' + str(name)+ '  Finish')
                break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted ..')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
