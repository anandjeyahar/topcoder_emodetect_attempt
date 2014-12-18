import csv
import argparse
import logging
import os
import cv2
import cv
import numpy
import sys
sys.path.append('/home/anand/Downloads/devbox_configs/')
import backend
from backend import fdmod
import json

logger = logging.getLogger()

parser = argparse.ArgumentParser(description='Run wishlist python flask app')
parser.add_argument('--port', dest='port', action=None,
                               type=int, default=8880,
                                                  help='port to run the app on ')
parser.add_argument('--folder', dest='outFolder', type=str,
                        default='./', help='Folder where to dump the edge detected images')

parser.add_argument('--edge-detect', dest='edgeDetect', default=False,
                    help='create new edge detected images')

parser.add_argument('--no-edge-detect', dest='noEdgeDetect',
                    help='create new edge detected images')

parser.add_argument('--input-file', dest='inpFile', type=str, default='./example_training.txt',
                    help='training data file')


trainingData = dict()       # Dict of the form {<imagefilename>: [list of bools for emotion, categories below]
featureData = dict()
emotionCategories = ['angry', 'anxious', 'confident', 'happy', 'neutral', 'sad', 'surprised']
CANNY_LOW_THRESH = 10.0
CANNY_HIGH_THRESH = 15.0
def int_ify(lst):
    lst1 = list()
    for each in lst:
        lst1.append(int(each))
    return lst1

def main(args):
    with open(args.inpFile, 'rb') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            if reader.line_num == 1:
                header = row
                continue
            intList = int_ify(row[1:])
            if sum(intList) != 1:
               logger.warn('invalid row in data file %s '%(str(row)))
               continue
            else:
                trainingData[row[0]] = intList

    FD = fdmod.FeatureDetect()
    if args.edgeDetect:
        if not os.path.exists(os.path.join(args.outFolder, 'data', 'edges')):
            os.makedirs(os.path.join([args.outFolder, 'data','edges']))
        for key in trainingData.keys():
            image = cv.LoadImage(os.path.join('./','data',key), cv.CV_LOAD_IMAGE_COLOR)
            outImageFile = os.path.join(args.outFolder, 'data', 'edges', key)
            edgeImage = cv.CreateImage((250, 250), 8,1)
            grayImage = cv.CreateImage((250, 250), 8,1)
            cv.CvtColor(image, grayImage, cv.CV_BGR2GRAY)
            cv.Canny(grayImage, edgeImage, CANNY_LOW_THRESH, CANNY_HIGH_THRESH)
            cv2.imwrite(outImageFile, numpy.asarray(edgeImage[:,:]))

    for key in trainingData.keys():
        image = cv2.imread(os.path.join('./','data',key), cv.CV_LOAD_IMAGE_COLOR)
        FD.image = image
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        FD.grayImage = grayImage
        FD.detectFace()
        FD.detectEyes()
        FD.detectLips()
        featureData.update({key:FD.features})

    with open('calculated_data.json', 'wb') as out_fd:
        out_fd.write(json.dumps(featureData))

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

