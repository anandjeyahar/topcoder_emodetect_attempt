import argparse
import sys
sys.path.append('/home/anand/Downloads/devbox_configs/')
import backend
from backend import fdmod
import json
import logging
import cv2
import cv
import os
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
    featureData.get(imgFile).update({'emotion':trainingData.get(imgFile)})
    return featureData

def getCannyEdges(imgFile):

    image = cv.LoadImage(os.path.join('./', 'data', imgFile), cv.CV_LOAD_IMAGE_COLOR)
    outImageFile = os.path.join(args.outFolder, 'data', 'edges', imgFile)
    edgeImage = cv.CreateImage((250, 250), 8,1)
    grayImage = cv.CreateImage((250, 250), 8,1)
    cv.CvtColor(image, grayImage, cv.CV_BGR2GRAY)
    cv.Canny(grayImage, edgeImage, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
    cv2.imwrite(outImageFile, numpy.asarray(edgeImage[:,:]))

def main(args):

    with open(args.inputFile, 'rb') as calc_fd:
        calcAreas = json.load(calc_fd)
    with open('./trainingData.json', 'rb') as inp_fd:
        trainingData = json.load(inp_fd)
    for key in [trainingData.keys()[3]]:
        calcAreas = getImageAreas(key)
        print calcAreas.get(key), key
        if not calcAreas.get(key).get('eyeCorners'):
            logging.warn('No eyeCorners for %s', % key)
            break
        if not calcAreas.get(key).get('faceCorners'):
            logging.warn('No eyeCorners for %s', % key)
            break
        # top-left corner is 0,0 so forehead is above eyes, therefore starts at min()
        forehead = ((min(calcAreas.get(key).get('eyeCorners')[0], calcAreas.get(key).get('faceCorners')[0]),
                    max(calcAreas.get(key).get('eyeCorners')[1], calcAreas.get(key).get('faceCorners')[1]),),
                    (max(calcAreas.get(key).get('eyeCorners')[2], calcAreas.get(key).get('faceCorners')[2]),
                        min(calcAreas.get(key).get('eyeCorners')[3], calcAreas.get(key).get('faceCorners')[3])))
        print forehead

    foreheadArea = () # higher co-ordinates of eye corners and higher co-ordinates of face.
    # Count the edges in the area(forehead) above the eyes
    # Vertical/edges perpendicular to eyes ==> confusion/anxiety
    # Count the edges in the area(cheeks) around the mouth
    # Count the edges in the area(eye sockets) around the eyes(edges parallel to eye shape ==> happy
    #
    # find reasonable weighted sum of these three counts to form a measure that cna classify into the seven categories of emoiton
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
