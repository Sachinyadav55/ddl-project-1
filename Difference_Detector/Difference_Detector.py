import cv2 as cv
import numpy as np
import argparse
from os import listdir
from os.path import isfile, join
import time



parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(usage="python3 Difference_Detector.py <Path File> \nThis program is a naive implementation of the Difference Detector described in the Noscope paper", description="")
parser.add_argument('-v', '--version', action='version',version='%(prog)s 1.0', help="Show program's version number and exit.")
parser.add_argument('DirectoryofImages',type =str, help='Path of the folder containing the images')
parser.add_argument('-t','--Thres', help='Threshold of the image difference', default=0.3)
args = parser.parse_args()


index=0
Directory = args.DirectoryofImages
onlyfiles = [f for f in listdir(Directory) if isfile(join(Directory, f))]
onlyfiles.sort()

#print(onlyfiles)

def Difference_Detector_screens(Image1,Image2):
    """
    This is a prototype function is to analyze two images side by side
    Shows a difference detector screen
    """
    firstImage = cv.imread(Image1)
    secondImage = cv.imread(Image2)

    resizeFirstImage = cv.resize(firstImage, None, fx=0.3, fy=0.3)
    image = resizeFirstImage
    resizeSecondImage = cv.resize(secondImage, None, fx=0.3, fy=0.3)
    numpy_horizontal_concat = np.concatenate((image, resizeSecondImage), axis=1)
    winName = 'Image Comparision'
    cv.namedWindow(winName)
    cv.moveWindow(winName, 40, 30)
    cv.imshow(winName, numpy_horizontal_concat)
    cv.waitKey()

    imageShape1 = resizeFirstImage.shape
    imageShape2 = resizeSecondImage.shape

    h1 = imageShape1[0]
    w1 = imageShape1[1]

    h2 = imageShape2[0]
    w2 = imageShape2[1]

    if h1 != h2 or w2 != w2:
        print("Size of images are different")
    else:
        for y in range(0, h1):
            for x in range(0, w1):
                if resizeFirstImage[y, x][0] != resizeSecondImage[y, x][0] or \
                        resizeFirstImage[y, x][1] != resizeSecondImage[y, x][1] or \
                        resizeFirstImage[y, x][2] != resizeSecondImage[y, x][2]:
                    
                    resizeSecondImage[y, x][0] = 0
                    resizeSecondImage[y, x][1] = 0
                    resizeSecondImage[y, x][2] = 255

    winName = 'Difference Detection'
    cv.namedWindow(winName)
    cv.moveWindow(winName, 40, 30)
    cv.imshow(winName, resizeSecondImage)
    cv.waitKey()
    #cv.destroyAllWindows()
    return None




def Difference_Detector(Image1,Image2):
    """
    Naive Implementation of Difference Detector
    Input: Two image: Image1, Image2
    Output: Matrix showing the difference between two images
    """
    firstImage = cv.imread(Image1)
    secondImage = cv.imread(Image2)
    
    resizeFirstImage = cv.resize(firstImage, None, fx=0.3, fy=0.3)
    resizeSecondImage = cv.resize(secondImage, None, fx=0.3, fy=0.3)

    imageShape1 = resizeFirstImage.shape
    imageShape2 = resizeSecondImage.shape
    Distance_Matrix = np.zeros(resizeFirstImage.shape)
    #print(Distance)
    h1 = imageShape1[0]
    w1 = imageShape1[1]

    h2 = imageShape2[0]
    w2 = imageShape2[1]

    if h1 != h2 or w2 != w2:
        print("Size of images are different")
    else:
        for y in range(0, h1):
            for x in range(0, w1):
                if resizeFirstImage[y, x][0] != resizeSecondImage[y, x][0] or \
                        resizeFirstImage[y, x][1] != resizeSecondImage[y, x][1] or \
                        resizeFirstImage[y, x][2] != resizeSecondImage[y, x][2]:

                            Distance_Matrix[y,x][0] = resizeSecondImage[y,x][0]-resizeFirstImage[y,x][0]
                            Distance_Matrix[y,x][1] = resizeSecondImage[y,x][1]-resizeFirstImage[y,x][1]
                            Distance_Matrix[y,x][2] = resizeSecondImage[y,x][2]-resizeFirstImage[y,x][2]
                            
    #print(Distance_Matrix)

    return Distance_Matrix

while index <len(onlyfiles)-1:
    image1 = str(Directory)+str(onlyfiles[index])
    index+=1
    image2 = str(Directory)+str(onlyfiles[index])
    Difference_Detector(image1,image2)
