import numpy as np
import math
import cv2
from shapely.geometry import Point, Polygon
from scipy.spatial import distance


### Global variable ###

# func rotate
TEXT_MIN_WIDTH = 15      # min reduced px width of detected text contour
TEXT_MIN_HEIGHT = 2      # min reduced px height of detected text contour
TEXT_MIN_ASPECT = 1.5    # filter out text contours below this w/h ratio
TEXT_MAX_THICKNESS = 10  # max reduced px thickness of detected text contour
TEXT_MAX_HEIGHT = 100 
########################

def convertBinaryBlurThresh(picture,blurWidth=5,blurHeight=5):
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurWidth,blurHeight), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def findContours(picture, parameterRETR,parameterChain):
    cnts = cv2.findContours(picture, parameterRETR, parameterChain)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    return cnts

def findDistance(box,w,h):
    #point of box cnt
    x1=box[0][0]
    y1=box[0][1]
    x2=box[1][0]
    y2=box[1][1]
    x3=box[2][0]
    y3=box[2][1]
    x4=box[3][0]
    y4=box[3][1]

    #find distance between point for find width and height
    d = distance.euclidean((x1,y1), (x2,y2))
    d2 = distance.euclidean((x1,y1), (x4,y4))
    if d>d2:
        dlong=d
        xlong=[x1,x2]
        ylong=[y1,y2]
        dshort=d2
        xshort=[x1,x4]
        yshort=[y1,y4]
    else:
        dlong=d2
        xlong=[x1,x4]
        ylong=[y1,y4]
        dshort=d
        xshort=[x1,x2]
        yshort=[y1,y2]
    if w>h:
        width = dlong
        height = dshort
        return height,width,xlong,ylong
    else:
        width = dshort
        height = dlong
        return height,width,xshort,yshort

def removeLine(picture):
    #first dilate
    thresh = convertBinaryBlurThresh(picture)

    # set kernel for remove horizontal line & vertical line
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,25))
    detected_lines2 = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

    #find contours & remove line with white color 
    cntsHorizontal = findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsVertical = findContours(detected_lines2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cntsHorizontal:
        cv2.drawContours(picture, [c], -1, (255,255,255), -1)
    for c in cntsVertical:
        cv2.drawContours(picture, [c], -1, (255,255,255), -1)
    return picture

def findAngle(x1,y1,x2,y2):
    angleCal = math.degrees(math.atan2(y2-y1,x2-x1))
    return angleCal

def findAverageAngle(angle):
    for key,val in enumerate(angle):
        if val > 135.0:
            angle[key] = 180-val
        elif 135> val > 45:
            angle[key]=90-val
        elif val<-135:
            angle[key]=180+val
        elif val<-45:
            angle[key]=90+val
        else:
            angle[key]=val
    angleAvg = (sum(angle)/len(angle))
    return angleAvg

def removePicture(externalBox, imgNotText, image):
    for exBox in externalBox:
        cv2.drawContours(imgNotText,[exBox],-1,(0,0,0),-1)

    imgNotText = cv2.cvtColor(imgNotText, cv2.COLOR_BGR2GRAY)
    deleteImage = findContours(imgNotText, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for delIm in deleteImage:
        cv2.drawContours(image,[delIm],-1,(255,255,255),-1)
    return image

def skipPage(image):
    #skip image or difficult bg
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = gray.flatten().tolist()
    (unique, counts) = np.unique(flat, return_counts=True)
    if max(counts)/len(flat)*100 < 10:
        return True
    return False

def prepareRotated(picture):
    image = picture.copy()
    imageOCR = convertBinaryBlurThresh(image)

    #use dilate for create externalCnt | erode for 
    kernalDilate = cv2.getStructuringElement(cv2.MORPH_RECT,(11,5))
    kernalErode = cv2.getStructuringElement(cv2.MORPH_RECT,(21,3))
    dilate = cv2.dilate(imageOCR,kernalDilate,iterations=3)
    erode = cv2.erode(dilate,kernalErode,iterations=3)
    h,w,c = image.shape

    externalCnts = findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    internalCnts = findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    imgNotText = image.copy()
    imgNotText = cv2.rectangle(imgNotText, (0,0), (w,h), (0,0,0),-1)
    
    internalText = []
    internalNotText = []
    externalCntBox = []

    #check internalCnt is picture or text
    for inCnt in internalCnts:
        rect = cv2.minAreaRect(inCnt)
        x,y,w,h = cv2.boundingRect(inCnt)
        box = np.int0(cv2.boxPoints(rect))

        #find heigth and width to check that cnt is picture or text
        height,width,xlow,ylow = findDistance(box,w,h)
        if height < TEXT_MIN_HEIGHT or width < TEXT_MIN_WIDTH or width < TEXT_MIN_ASPECT*height or height > TEXT_MAX_HEIGHT: 
            if height > TEXT_MAX_HEIGHT:
                internalNotText.append([(x+5,y+5),(x+w-5,y+5),(x+w-5,y+h-5),(x+5,y+h-5)])
                cv2.rectangle(imgNotText, (x+10, y+10), (x + w-10, y + h-10), (255,0,12),3)
            continue
        internalText.append([box,xlow,ylow])
    
    checksame=[]
    externalBox=[]
    angleBox=[]

    for exCnt in externalCnts:
        val=False
        boundaryBox = cv2.boundingRect(exCnt)
        rect = cv2.minAreaRect(exCnt)
        box = np.int0(cv2.boxPoints(rect))
        polygon = Polygon(box)
        for index, notText in enumerate(internalNotText):
            p0 = Point(notText[0])
            p1 = Point(notText[1])
            p2 = Point(notText[2])
            p3 = Point(notText[3])
            val = p1.within(polygon) and p2.within(polygon) and p3.within(polygon) and p0.within(polygon)
            if val:
                cv2.drawContours(imgNotText,[box],-1,(255,255,255),-1)
                internalNotText.pop(internalNotText.index(notText))
                break
            
        if not val:
            externalBox.append(box)
            externalCntBox.append(boundaryBox)
            angle = []
            avgAngle=0
            if len(internalText) == 0:
                angleBox.append(avgAngle)
            for indexText, textBox in enumerate(internalText):
                pi0 = Point(textBox[0][0])
                pi1 = Point(textBox[0][1])
                pi2 = Point(textBox[0][2])
                pi3 = Point(textBox[0][3])
                vali = pi1.within(polygon) and pi2.within(polygon) and pi3.within(polygon) and pi0.within(polygon)
                if vali:
                    angleSingle = findAngle(textBox[1][0],textBox[2][0],textBox[1][1],textBox[2][1])
                    angle.append(angleSingle)
                    cv2.arrowedLine(picture,(textBox[1][0],textBox[2][0]),(textBox[1][1],textBox[2][1]),(0,255,0),2)
                if len(internalText)-1 == indexText:
                    if(len(angle) != 0):
                        avgAngle = findAverageAngle(angle)
                    angleBox.append(avgAngle)

    # resize('testang',picture)
    image = removePicture(externalBox, imgNotText, image)     
    return image,angleBox,externalCntBox

def resize(name,picture):
    #resize 
    scale_percent = 50
    width = int(picture.shape[1] * scale_percent / 100)
    height = int(picture.shape[0] * scale_percent / 100)
    #dsize
    dsize = (width, height)
    if width != 0 and height != 0: 
        pictureResize = cv2.resize(picture, dsize)
        cv2.imshow(name,pictureResize)
        cv2.waitKey() 

def rotated(img,angle,inx,box):
    x,y,w,h = box
    center = (w / 2, h / 2)
    scale = 1.0
    cropToOCR = img[y:y+h+5, x:x+w].copy()
    M = cv2.getRotationMatrix2D(center, angle[inx], scale)
    rotated = cv2.warpAffine(cropToOCR, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # resize('rotate',rotated)
    return rotated