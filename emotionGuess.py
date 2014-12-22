import argparse
import sys
import facedetect as fdmod
sys.path.append('/home/anand/Downloads/devbox_configs/')
from backend import mine as c45Miner
import json
import logging
import cv2
import cv
import csv
import os
import numpy
import statsmodels

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

def count_edges(edgeImageArray, pixelRange, nsamples=1):
    """
    Function counts edges/pixels within the given range, with 255 value
    pixelRange: is a tetra-ple of x1, y1, x2,y2
    """
    # TODO; take count at multiple axes and average
    x1, y1, x2, y2 = pixelRange
    verticalEdges = dict()
    horizontalEdges = dict()
    pivotX = (x1 + x2) / (nsamples + 1)
    pivotY = (y1 + y2) / (nsamples + 1)
    for y in range(y1, y2):
        if edgeImageArray[pivotX][y] == 255:
            if verticalEdges.get(pivotX):
                verticalEdges[pivotX] += 1
            else:
                verticalEdges[pivotX] = 1

    for x in range(x1, x2):
        if edgeImageArray[x][pivotY] == 255:
            if horizontalEdges.get(pivotY):
                horizontalEdges[pivotY] += 1
            else:
                horizontalEdges[pivotY] = 1
    return {'vertical': numpy.mean(verticalEdges.values()), 'horizontal': numpy.mean(horizontalEdges.values())}

def offline_train(args):

    # Count the edges in the area(forehead) above the eyes
    # Vertical edges perpendicular to eyes on the eyesides > 5==> happy/confident/neutral
    # Vertical/edges perpendicular to eyes on the forehead ==> confusion/anxiety
    # horizontal edger parallel to mouth on the chin > 15 ==> happy/neutral/confident
    # Count the edges in the area(cheeks) around the mouth
    # Count the edges in the area(eye sockets) around the eyes(edges parallel to eye shape ==> happy
    # find reasonable weighted sum of these three counts to form a measure that cna classify into the seven categories of emoiton
    # write to csv imgName, foreheadEdges(ho), forehadedges(v), ,,,,emotion

#    with open(args.inputFile, 'rb') as calc_fd:
#        calcAreas = json.load(calc_fd)
    with open('./trainingData.json', 'rb') as inp_fd:
        trainingData = json.load(inp_fd)
    c45Input = dict()
    c45Result = list()
    allEdgeCounts = dict()
    fields = ['filename', 'foreheadHorizEdges', 'foreheadVertEdges',
                'eyeSidesHorizEdges', 'eyeSidesVertEdges',
                'lipSidesHorizEdges', 'lipSidesVertEdges',
               'chinHorizEdges', 'chinVertEdges', 'emotion']
    with open('allFeaturesData.csv', 'wb') as csv_fd:
        csvWriter = csv.DictWriter(csv_fd, fieldnames=fields)
        csvWriter.writeheader()

        for key in trainingData.keys():
            eyeSidesEdges = dict()
            lipSidesEdges = dict()
            calcAreas = getImageAreas(key)
            calcAreas.get(key).update({'emotion':trainingData.get(key)})
            if not calcAreas.get(key).get('eyeCorners'):
                logging.warn('No eyeCorners for %s'% key)
                continue
            if not calcAreas.get(key).get('faceCorners'):
                logging.warn('No faceCorners for %s'% key)
                continue
            # top-left corner is 0,0 so forehead is above eyes, therefore starts at min()
            forehead = (calcAreas.get(key).get('eyeCorners')[2], calcAreas.get(key).get('faceCorners')[1],
                        calcAreas.get(key).get('eyeCorners')[2], calcAreas.get(key).get('eyeCorners')[1])
            eyeSidesLeft = (calcAreas.get(key).get('eyeCorners')[0] - 20, calcAreas.get(key).get('eyeCorners')[1],
                         calcAreas.get(key).get('eyeCorners')[0], calcAreas.get(key).get('eyeCorners')[3])
            eyeSidesRight = (calcAreas.get(key).get('eyeCorners')[2], calcAreas.get(key).get('eyeCorners')[1],
                            calcAreas.get(key).get('eyeCorners')[2] + 20, calcAreas.get(key).get('eyeCorners')[3])
            if not eyeSidesLeft <= calcAreas.get(key).get('eyeCorners'):
                logging.warn("%s image eyes are messed up %s" %(key, str(calcAreas.get(key).get('eyeCorners'))))
            assert eyeSidesRight >= calcAreas.get(key).get('eyeCorners')
            if not calcAreas.get(key).get('lipCorners'):
                logging.warn('No mouthCorners for %s'% key)
                continue
            lipSidesLeft = (calcAreas.get(key).get('lipCorners')[0] - 20, calcAreas.get(key).get('lipCorners')[1],
                            calcAreas.get(key).get('lipCorners')[0], calcAreas.get(key).get('lipCorners')[3])
            lipSidesRight = (calcAreas.get(key).get('lipCorners')[2], calcAreas.get(key).get('lipCorners')[1],
                            calcAreas.get(key).get('lipCorners')[2] + 20, calcAreas.get(key).get('lipCorners')[3])
            chin =(calcAreas.get(key).get('lipCorners')[2], calcAreas.get(key).get('lipCorners')[1],
                    calcAreas.get(key).get('lipCorners')[0], calcAreas.get(key).get('faceCorners')[3])

            # Get the edgeImage and count the edges.

            edgeImage = getCannyEdges(key)
            eyeSidesRightEdges = count_edges(edgeImage, eyeSidesRight)
            for k, v in count_edges(edgeImage, eyeSidesLeft).iteritems():
                eyeSidesEdges[k] = v + eyeSidesRightEdges.get(k)
            lipSidesRightEdges = count_edges(edgeImage, lipSidesRight)
            for k, v in count_edges(edgeImage, lipSidesLeft).iteritems():
                lipSidesEdges[k] = v + lipSidesRightEdges.get(k)

            allEdgeCounts[key] = {'foreheadEdges': count_edges(edgeImage, forehead),
                                    'eyeSidesEdges': eyeSidesEdges,
                                    'lipSidesEdges': lipSidesEdges,
                                    'chinEdges': count_edges(edgeImage, chin)}

            row = {'filename': key, 'foreheadHorizEdges': allEdgeCounts.get(key).get('foreheadEdges').get('horizontal'),
                    'foreheadVertEdges': allEdgeCounts.get(key).get('foreheadEdges').get('vertical'),
                    'eyeSidesHorizEdges': eyeSidesEdges.get('horizontal'),
                    'eyeSidesVertEdges': eyeSidesEdges.get('vertical'),
                    'lipSidesHorizEdges': lipSidesEdges.get('horizontal'),
                    'lipSidesVertEdges': lipSidesEdges.get('vertical'),
                    'chinHorizEdges': allEdgeCounts.get(key).get('chinEdges').get('horizontal'),
                    'chinVertEdges': allEdgeCounts.get(key).get('chinEdges').get('vertical'),
                    'emotion': emotionCategories[trainingData.get(key).index(1)]
                    }
            csvWriter.writerow(row)
            c45Result.append(row.pop('emotion'))
            for k, v in row.iteritems():
                if not c45Input.get(k):
                    c45Input[k] = [v]
                else:
                    c45Input[k].append(v)
    # decisionTree = c45Miner.mine_c45(c45Input, c45Result)
    # rules = c45Miner.tree_to_rules(decisionTree)
    # with open('c45Rules', 'wb') as out_fd:
    #     out_fd.write(json.dumps(rules))
    with open('allEdgeCounts.json', 'wb') as out_fd:
        out_fd.write(json.dumps(allEdgeCounts))

def training(imgArray, emotions):
    pass

def testing(imgArray):
    pass

if __name__ == '__main__':
    args = parser.parse_args()
    offline_train(args)
