import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import pickle

MIN_MATCHES = 10

class CamCompass(object):
    ''' This class implements a visual compass, or
    in other words, a compass based on images
    from a webcam. A few images have to be fed
    in advance, and their corresponding angle.
    Later any provided image can have their SIFT
    features compared with the database images
    to find the best match and corresponding
    angle.
    '''
    database = [] # Database for consultation
    sift = None # SIFT detector object
    flann = None # Flann matcher object
    def __init__(self):
        pass
    def registerFile(self, filename, angle):
        ''' This method registers a single
        file to the database and its associate
        angle.
        '''
        print 'Registering file',filename
        # Create the SIFT detector
        self.sift = cv2.xfeatures2d.SIFT_create()
        # Create the point cloud matcher
        index_params = dict(algorithm = 0, trees = 5)
        search_params = dict(checks = 50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        # Find the SIFT points in the ref_imgs
        self._calculateSifts(filename, angle)
    def registerFiles(self, filelist, angle):
        ''' This method registers a list of
        files to the database, with a single
        angle for all of them.
        '''
        for filename in filelist:
            self.registerFile(filename, angle)
    def registerFolder(self, folderpath, angle):
        ''' Convenience method to add all
        files in a folder. All .png files
        foudn in that folder will be added
        using the same angle reference.
        '''
        allfiles = [f for f in listdir(folderpath) if isfile(join(folderpath, f))]
        for filename in allfiles:
            if len(filename) > 3:
                if filename[-3:] == 'png':
                    self.registerFile(join(folderpath,filename), angle)
    def _calculateSifts(self, img, angle):
        ''' This method is called by the
        constructor. It finds the SIFT
        feature points of the reference
        image and stores them for later
        reference
        '''
        mat = cv2.imread(img,0)
        kp, des = self.sift.detectAndCompute(mat,None)
        self.database.append((kp, des, angle))
    def getAngle(self, img_mat):
        '''This is the main method. The algorithm
        will try to match the SIFT features of
        the provided image to the ones in the
        database, and if a good match is found,
        the associated angle is returned
        '''
        kp2, des2 = self.sift.detectAndCompute(img_mat, None)
        results = []
        for kp1, des1, angle1 in self.database:
            matches = self.flann.knnMatch(des1, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.7*n.distance:
                    good.append(m)
            results.append((kp1, des1, angle1, good))
        kp1, des1, angle1, good = max(results, key=lambda x: x[3])
        if len(good) > MIN_MATCHES:
            return angle1
        else:
            return None
    def saveToFile(self, filename):
        ''' Saves the computed database of SIFT
        features and angles to a file. This can
        be used to save time.
        '''
        f = open(filename, 'w')
        pickle.dump(f, self.database)
        f.close()
    def readFromFile(self, filename):
        ''' Reads a pre-computed database of SIFT
        features and angles from a file.
        '''
        f = open(filename, 'r')
        self.database = pickle.load(f)
        f.close()

