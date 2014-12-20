import argparse
import sys
import facedetect as fdmod
import json
import logging
import cv2
import cv
import os
import numpy
logger = logging.getLogger()
parser = argparse.ArgumentParser(description='Run wishlist python flask app')

parser.add_argument('--inp-file', dest='inputFile', action=None,
                               type=str, default='./calculated_data.json',
                                                  help='file cto read data from ')

parser.add_argument('--train-file', dest='trainingFile', type=str,
                    default='./example_training.txt',
                    help='training data file')

trainingData = dict()       # Dict of the form {<imagefilename>: [list of bools for emotion, categories below]
emotionCategories = ['angry', 'anxious', 'confident', 'happy', 'neutral', 'sad', 'surprised']
CANNY_LOW_THRESH = 5.0
CANNY_HIGH_THRESH = 15.0

allEdgeCounts = dict()

def getImageAreas(imgFile):
    featureData = dict()
    FD = fdmod.FeatureDetect()
    image = cv2.imread(os.path.join('./', 'data', imgFile), cv.CV_LOAD_IMAGE_COLOR)
    FD.image = image
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    FD.grayImage = grayImage
    FD.detectFace()
    FD.detectEyes()
    FD.detectLips()
    featureData.update({imgFile:FD.features})
    return featureData

def getCannyEdges(imgFile):

    image = cv.LoadImage(os.path.join('./', 'data', imgFile), cv.CV_LOAD_IMAGE_COLOR)
    #outImageFile = os.path.join(args.outFolder, 'data', 'edges', imgFile)
    edgeImage = cv.CreateImage((250, 250), 8,1)
    grayImage = cv.CreateImage((250, 250), 8,1)
    cv.CvtColor(image, grayImage, cv.CV_BGR2GRAY)
    cv.Canny(grayImage, edgeImage, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    return numpy.asarray(edgeImage[:,:])

def count_edges(edgeImageArray, pixelRange):
    """
    Function counts edges/pixels within the given range, with 255 value
    pixelRange: is a tetra-ple of x1, y1, x2,y2
    """
    # TODO; take count at multiple axes and average
    x1, y1, x2, y2 = pixelRange
    verticalEdges = 0
    horizontalEdges = 0
    midX = (x1 + x2) / 2
    midY = (y1 + y2) / 2
    for y in range(y1, y2):
        if edgeImageArray[midX][y] == 255:
            verticalEdges += 1

    for x in range(x1, x2):
        if edgeImageArray[x][midY] == 255:
            horizontalEdges += 1
    return {'vertical': verticalEdges, 'horizontal': horizontalEdges}

def main(args):

    with open(args.inputFile, 'rb') as calc_fd:
        calcAreas = json.load(calc_fd)
    with open('./trainingData.json', 'rb') as inp_fd:
        trainingData = json.load(inp_fd)

    for key in trainingData.keys():
        calcAreas = getImageAreas(key)
        calcAreas.get(key).update({'emotion':trainingData.get(key)})
        if not calcAreas.get(key).get('eyeCorners'):
            logging.warn('No eyeCorners for %s'% key)
            continue
        if not calcAreas.get(key).get('faceCorners'):
            logging.warn('No faceCorners for %s'% key)
            continue
        if not calcAreas.get(key).get('lipCorners'):
            logging.warn('No mouthCorners for %s'% key)
            continue
        # top-left corner is 0,0 so forehead is above eyes, therefore starts at min()
        forehead = (min(calcAreas.get(key).get('eyeCorners')[0], calcAreas.get(key).get('faceCorners')[0]),
                    max(calcAreas.get(key).get('eyeCorners')[1], calcAreas.get(key).get('faceCorners')[1]),
                    max(calcAreas.get(key).get('eyeCorners')[2], calcAreas.get(key).get('faceCorners')[2]),
                        min(calcAreas.get(key).get('eyeCorners')[3], calcAreas.get(key).get('faceCorners')[3]))
        edgeImage = getCannyEdges(key)
        allEdgeCounts[key] = {'foreheadEdges': count_edges(edgeImage, forehead)}
        mouthArea = ()
    # Count the edges in the area(forehead) above the eyes
    # Vertical/edges perpendicular to eyes ==> confusion/anxiety
    # Count the edges in the area(cheeks) around the mouth
    # Count the edges in the area(eye sockets) around the eyes(edges parallel to eye shape ==> happy
    # find reasonable weighted sum of these three counts to form a measure that cna classify into the seven categories of emoiton
    with open('allEdgeCounts.json', 'wb') as out_fd:
        out_fd.write(json.dumps(allEdgeCounts))
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
