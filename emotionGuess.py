import argparse
import sys
sys.path.append('/home/anand/Downloads/devbox_configs/')
import backend
import json
import logging
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


def main(args):
    
    with open(args.inputFile, 'rb') as calc_fd:
        calcAreas = json.load(calc_fd)
    print calcAreas

    foreheadArea = () # higher co-ordinates of eye corners and higher co-ordinates of face.
    # Count the edges in the area(forehead) above the eyes
    # Count the edges in the area(cheeks) around the mouth
    # Count the edges in the area(eye sockets) around the eyes
    # find reasonable weighted sum of these three counts to form a measure that cna classify into the seven categories of emoiton
    pass
    
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
