import feature_extraction as fe
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io, color
import numpy as np
import matplotlib.pyplot as plt


class Feature:
    SP_N_SEGMENTS = 300
    SP_COMPACTNESS = 30
    SP_SIGMA = 1

    def __init__(self):
        self.image = []
        self.image_greyscale = []
        self.label_image = []
        self.superpixel = []
        self.superpixel_labels = []
        # Features to extract from superpixel
        self.superpixel_location = []
        self.superpixel_color = []
        self.superpixel_size = []
        self.superpixel_hog = []
        self.superpixel_color_histogram = []
        self.superpixel_texture = []

    def create(self,image_url,label_url):
        self.loadImage(image_url)
        #self.loadLabelImage(label_url)
        self.extractFeatures()

    def loadImage(self, image_url):
        self.image = img_as_float(io.imread(image_url))
        self.image_greyscale = color.rgb2gray(self.image)
        self.superpixel = slic(self.image, n_segments=Feature.SP_N_SEGMENTS, compactness=Feature.SP_COMPACTNESS, sigma=Feature.SP_SIGMA)

    def loadLabelImage(self, label_url):
        self.label_image = io.imread(label_url)
        self.superpixel_labels = fe.getSuperPixelLabel(self.superpixel, self.label_image, 0.5)

    def extractFeatures(self):
        self.superpixel_location = fe.getSuperPixelLocation(self.superpixel)
        #self.superpixel_color = fe.getSuperPixelMeanColor(self.superpixel, self.image)
        self.superpixel_hog = fe.getSuperPixelOrientedHistogram(self.superpixel, self.image)
        self.superpixel_size = fe.getSuperPixelSize(self.superpixel)
        self.superpixel_texture = fe.getSuperPixelTexture(self.superpixel, self.image)
        #self.featureVectors = np.vstack((self.superpixel_location.T, self.superpixel_color.T, self.superpixel_hog.T, self.superpixel_size.T, self.superpixel_texture.T)).T

    def getSuperpixelsLocation(self):
        return self.superpixel_location

    def getSuperpixelsColor(self):
        return self.superpixel_color

    def getSuperpixelsHog(self):
        return self.superpixel_hog

    def getSuperpixelsSize(self):
        return self.superpixel_size

    def getSuperpixelsTexture(self):
        return self.superpixel_texture

    def getFeaturesVectors(self):
        return self.featureVectors

    def getSuperpixelLabels(self):
        return self.superpixel_labels

    def getImage(self):
        return self.image

    def getSuperpixelImage(self):
        return self.superpixel

    def getLabelImage(self):
        return self.label_image

    # DEBUG FUNCTIONS
    def plot(self,image):
        plt.imshow(image)
        plt.show()

    def plotGreyscale(self,image):
        plt.imshow(image,cmap=plt.cm.gray, interpolation='nearest',vmin=0, vmax=255)
        plt.show()

    def showSuperpixel(self):
        self.plot(mark_boundaries(self.image, self.superpixel))

    def showGreyscale(self):
        self.plotGreyscale(self.image_greyscale*256)

    def showImage(self):
        self.plot(self.image)

    def showLabels(self):
        image = np.copy(self.image)
        new_image = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        num_superpixels = np.max(self.superpixel) + 1
        for i in range(0,num_superpixels):
            indices = np.where(self.superpixel==i)
            label = '1' if self.superpixel_labels[i] else '0'
            new_image[indices] = [0,label,0]
            indices = np.where(new_image > 0.5)
            image[indices] = 1
        self.plot(image)