import numpy as np
import matplotlib.image as mpimg
from skimage.segmentation import slic
from sklearn.feature_extraction import image
from skimage import exposure
from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
from skimage import color

# @input 
#      img: original image
#      n_segments: number of superpixels
#      compactness: indicator of compactness
# @output
#      superpixels: 2D array segmentation labels
def getSuperpixels(img, n_segments, compactness):
    superpixels = slic(img, n_segments=300, compactness=20, sigma=1)
    return superpixels


# @input 
#      superpixels: 2D array segmentation 
# @output
#      superpixel_location: 2D array nx2 center location of all superpixels
def getSuperPixelLocation(superpixels):
    superpixel_location = []  
    for i in np.unique(superpixels):
        indices = np.where(superpixels == i)
        x = np.mean(indices[0])
        y = np.mean(indices[1])
        superpixel_location.append([x,y])
    return np.array(superpixel_location)

# @input 
#      superpixels: 2D array segmentation 
# @output
#      size: 1D array superpixels' sizes
def getSuperPixelSize(superpixels):
    size = []
    for i in np.unique(superpixels):
        indices = np.where(superpixels == i)
        size.append(indices[0].shape)
    return np.array(size)

# @input 
#      superpixels: 2D array segmentation 
#      image: 3D array nxmx3 pixels color original image
# @output
#      mean_color: 2D array nx3 mean color of all superpixels
def getSuperPixelMeanColor(superpixels, image):
    mean_color = []
    for i in np.unique(superpixels):
        indices = np.where(superpixels==i)
        color = image[indices]
        mean_color.append([ np.mean(color[:,0]), np.mean(color[:,1]), np.mean(color[:,2])])
    return np.array(mean_color)

# @input 
#      superpixels: 2D array segmentation 
#      image: 3D array color original image
# @output
#      gradients: histogram of oriented gradient
def getSuperPixelOrientedHistogram(superpixels, image):
    grayImage = color.rgb2gray(image)
    gradients = []
    fd, hog_image = hog(grayImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    for i in np.unique(superpixels):
        indices = np.where(superpixels==i)
        gradient = [np.mean(hog_image_rescaled[indices])]
        gradients.append(gradient)
    return np.array(gradients)

# @input 
#      label_image: 3D array nxmx3 pixels color original image
# @output
#      label_pixel: pixel level labels
def getPixelLabel(label_image):
    label_pixel = label_image[:,:,2]
    return label_pixel

# @input 
#      superpixels: 2D array segmentation 
#      label_image: 3D array nxmx3 pixels color original image
#      thres: ratio threshold (0-1) for setting a superpixel to be true
# @output
#      labelSuperpixel: 1D Array label per super pixel (only 1 or 0)
def getSuperPixelLabel(superpixels, label_image, thres=0.5):
    labelAverage = []
    label_pixel = label_image[:,:,2]
    for i in np.unique(superpixels):
        indices = np.where(superpixels==i)
        labels = label_pixel[indices]
        protionTrue = 1.0*np.sum(labels)/len(labels)
        labelAverage.append(protionTrue)
    labelAverage = np.array(labelAverage)
    labelSuperpixel = np.array( labelAverage > thres)
    return np.array(labelSuperpixel)

# @input 
#      superpixels: 2D array segmentation 
#      image: 3D array nxmx3 pixels color original image
# @output
#      mean_color: 2D array nx2 texture (dissimilarity, correlation) of all superpixels
def getSuperPixelTexture(superpixels, image):
    texture = []
    numSuperpixels = np.max(superpixels) + 1
    greyImageq = np.around(color.rgb2gray(image) * 255, 0)
    greyImage = np.array(greyImageq, dtype='uint8')

    for i in range(0,numSuperpixels):
        indices = np.where(superpixels == i)
        glcm = greycomatrix([greyImage[indices]], [5], [0], 256, symmetric=True, normed=True)
        dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
        correlation = greycoprops(glcm, 'correlation')[0, 0]
        texture.append([dissimilarity, correlation])
    return np.array(texture)