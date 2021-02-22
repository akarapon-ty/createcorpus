#file and process
import os,sys,re
import multiprocessing as mp
import threading
import logging
import queue
import tensorflow as tf

# from another file
import tokenizer

# tf.get_logger().setLevel('ERROR')
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# physical_devices = tf.config.list_physical_devices('GPU')
# try:
#   # Disable all GPUS
#   tf.config.set_visible_devices([], 'GPU')
#   visible_devices = tf.config.get_visible_devices()
#   for device in visible_devices:
#     assert device.device_type != 'GPU'
# except:
#   # Invalid device or cannot modify virtual devices once initialized.
#   pass

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-12s %(processName)-8s %(message)s',
                    datefmt='%d-%m-%y %H:%M',
                    filename='./processLog.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(processName)-12s: %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def listDirectory(path):
    listDir = False
    try:
        listDir = [os.path.join(path,f) for f in os.listdir(path) if f.endswith(".txt")]
    except OSError:
        logging.info("Error: path Error maybe wrong name!!!!")
    return listDir

def readFile(path):
    with open(path, "r", encoding="utf8") as f:
        return f.read().split("\n\n") 
    return False

def createDirectory(path):
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s " % path)

def createFileCorpus(path):
    try:
        open(path, "x")
    except OSError:
        logging.info("Already Have Corpus file to update")

def writeFile(path, fullText):
    with open(path, "a", encoding='utf8') as f:
        for text in fullText:
            f.write(text)

def pipeLineProducer(path, queue, pathFolder, folderName):
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

    #### set variable to save file
    pageNumber = re.search('page-(.*).txt', path).group(1)
    pathSaveClean = pathFolder+"/page-" + pageNumber + ".txt"

    logging.info("FileName: " + str(folderName)+ " Page: "+pageNumber+ " Start Clean Process")
    fullTextInPage = readFile(path)
    createFileCorpus(pathSaveClean)
    fullTextClean = []
    for sentence in fullTextInPage:
        cleanSentence = tokenizer.cleanWord(sentence)
        cleanSentence = list(map(tokenizer.spellCheckAuto, cleanSentence))
        ### write file clean text
        fullTextClean.extend(cleanSentence)
        cleanSentence.append('\n')
        writeFile(pathSaveClean, cleanSentence)
    queue.put(fullTextClean)
    logging.info("FileName: " + str(folderName)+ " Page: "+pageNumber+ " Finish Clean Process")
    
def pipeLineConsumer(queueWriteFile, queueWork, path):
    while True:
        document = queueWriteFile.get()
        queueWork.put(1)
        writeFile(path, document)
        # logging.debug("Page:" + str(re.search('page-(.*).txt', path).group(1)) + " Finish write File")
        if queueWriteFile.empty():
            queueWork.get()

def main():
    ### plan
    ## customer (Thread) pull corpus in queue -> write file 
    ## walk(pathDir) -> mp -> pipeLineClean(pathFile)

    ## pipeLineClean -> readFile -> cleanWord --> return corpusInPage 
    ## add corpus to queue 

    #### set up name path variable 
    manager = mp.Manager()
    queueWriteFile = manager.Queue()
    queueWork = manager.Queue(1)
    pathCorpusFile ='./corpus.txt'
    pathCleanFileFolder ='./cleanOCR'

    #### create folder
    createFileCorpus(pathCorpusFile)
    createDirectory(pathCleanFileFolder)

    #### create consumer thread to write corpus file
    consumer = threading.Thread(target=pipeLineConsumer, args=(queueWriteFile,queueWork,pathCorpusFile, ))
    consumer.setDaemon=True
    consumer.start()

    while True:
        poolProducer = mp.Pool(processes=3)
        folderName = input(f'Enter folder name (only in path report!!)> ')
        #### set report ocr path
        path = '../KMUTT-Archives-Management-Django/KMUTTArchivesManagement/document-report/' +folderName
        listPathFileNames = listDirectory(path)
        #### set path & create to save clean text
        pathFolder = pathCleanFileFolder + '/' + folderName
        createDirectory(pathFolder)
        if not listPathFileNames:
            logging.info("Error: listPathFileNames return False")
            return
        for pathFileName in listPathFileNames:
            poolProducer.apply_async(pipeLineProducer, args=(pathFileName, queueWriteFile, pathFolder, folderName ))
        poolProducer.close()
        poolProducer.join()
        while True:
            if queueWork.empty():
                logging.info('FolderName: ' + str(folderName)+ '  Finish')
                break

if __name__ == '__main__':
   try:
     main()
   except KeyboardInterrupt:
     print ('\nInterrupted ..')
     try:
       sys.exit(0)
     except SystemExit:
       os._exit(0)