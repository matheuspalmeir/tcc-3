# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:00:39 2022

@author: gtvol
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import glob
import cv2
from feature import Feature
import feature_extraction as fe


df = gpd.read_file('C:/Users/gtvol/Downloads/campo/Dataset/Reference/CampoVerde_Oct2015_Jul2016.shp')
df.head()
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(15, 15))
df.plot(ax=ax)
'''
#print(df[['Oct_2015','Nov_2015','Dec_2015','Jan_2016','Feb_2016','Mar_2016','Apr_2016','May_2016','Jun_2016','Jul_2016']].apply(pd.Series.value_counts))
lenght = count_row = df[['Oct_2015','Nov_2015','Dec_2015','Jan_2016','Feb_2016','Mar_2016','Apr_2016','May_2016','Jun_2016','Jul_2016']].apply(pd.Series.value_counts).shape[0]
print(lenght)

'''


#from skimage.feature import graycomatrix, graycoprops
from skimage import data,io, color, img_as_ubyte
from skimage.feature import greycomatrix, greycoprops



# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2
'''
def segment_colorfulness(image, mask):
	# split the image into its respective RGB components, then mask
	# each of the individual RGB channels so we can compute
	# statistics only for the masked region
	(B, G, R) = cv2.split(image.astype("float"))
	R = np.ma.masked_array(R, mask=mask)
	G = np.ma.masked_array(B, mask=mask)
	B = np.ma.masked_array(B, mask=mask)
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`,
	# then combine them
	stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
	meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

# load the image in OpenCV format so we can draw on it later, then
# allocate memory for the superpixel colorfulness visualization
filename = 'C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif'
orig = cv2.imread(filename)
orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
print(np.max(orig))
print(np.min(orig))
vis = np.zeros(orig.shape[:2], dtype="float")
# load the image and apply SLIC superpixel segmentation to it via
# scikit-image
image = io.imread('C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif')
segments = slic(img_as_float(image), n_segments=100,
	slic_zero=True)

# loop over each of the unique superpixels
for v in np.unique(segments):
	# construct a mask for the segment so we can compute image
	# statistics for *only* the masked region
	mask = np.ones(image.shape[:2])
	mask[segments == v] = 0
	# compute the superpixel colorfulness, then update the
	# visualization array
	C = segment_colorfulness(orig, mask)
	vis[segments == v] = C
    
# to unsigned 8-bit integer array so we can use it with OpenCV and
# display it to our screen
vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
# overlay the superpixel colorfulness visualization on the original
# image
alpha = 0.6
overlay = np.dstack([vis] * 3)
output = orig.copy()
cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # show the output images
cv2.imshow("Input", orig)
cv2.imshow("Visualization", vis)
cv2.imshow("Output", output)
cv2.waitKey(0)
'''

import json
from osgeo import ogr
import geopandas
import rasterio
import rasterstats
from rasterio.features import rasterize
from rasterstats.io import bounds_window
def convert_poly_to_int(testpolywkt):
   # testpolywkt = 'POLYGON ((75694.8564184426 452182.812141684,75570.7757705624 451967.898155319,75322.6144748018 451967.898155319,75198.5338269215 452182.812141684,75322.6144748018 452397.726128049,75570.7757705624 452397.726128049,75694.8564184426 452182.812141684))'
    
    geom = ogr.CreateGeometryFromWkt(testpolywkt)
    
    coordinates = json.loads(geom.ExportToJson())['coordinates'][0]
    coordinates_rounded = [[round(pair[0]), round(pair[1])] for pair in coordinates]
    
    # Create ring
    ring = ogr.Geometry(ogr.wkbLinearRing)
    print('loop convert poly')
    for pair in coordinates_rounded:
        ring.AddPoint_2D(pair[0], pair[1])
       
    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    
    #print(poly.ExportToWkt())
    return poly.ExportToWkt()

def features_glcm(image_array):
    feat = []
    #img = io.imread(filename)
    
   # gray = color.rgb2gray(img)
    
   # image_array = np.array(gray, dtype='uint8')
    
    
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    matrix_coocurrence = greycomatrix(image_array, [3], angles, normed=False, symmetric=False)
    print(matrix_coocurrence.shape)
    contrast = greycoprops(matrix_coocurrence, 'contrast')
    homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    correlation = greycoprops(matrix_coocurrence, 'correlation')
    mean = np.apply_over_axes(np.mean, matrix_coocurrence, axes=(0, 1))[0, 0]  

    return [*contrast,*homogeneity,*correlation,*mean]


def image_process(img, feat):
    return img*feat




month = ['Oct_2015','Nov_2015','Dec_2015','Jan_2016','Feb_2016','Mar_2016','Apr_2016','May_2016','Jun_2016','Jul_2016']
classes = np.unique(df[month])
#print(classes)
class_dict = dict(zip(classes, range(len(classes))))

# a custom function for getting each value from the raster
def all_values(x):
    return x
    

def feat(img,df, month):
    # this larger cell reads data from a raster file for each training vector
    X_raw = []
    y_raw = []
    i = 0
    with rasterio.open(img, 'r') as src:
        for (label, geom) in zip(df[month], df.geometry):
            print(i)
            i+=1
            # read the raster data matching the geometry bounds
            window = bounds_window(geom.bounds, src.transform)
            # store our window information
            window_affine = src.window_transform(window)
            fsrc = src.read(window=window)
            # rasterize the geometry into the larger shape and affine
            mask = rasterize(
                [(geom, 1)],
                out_shape=fsrc.shape[1:],
                transform=window_affine,
                fill=0,
                dtype='uint8',
                all_touched=True
            ).astype(bool)
           
            # for each label pixel (places where the mask is true)
            label_pixels = np.argwhere(mask)
            
            for (row, col) in label_pixels:
                # add a pixel of data to X
               
                data = fsrc[:,row,col]
                one_x = np.nan_to_num(data, nan=1e-3)
                X_raw.append(one_x)
                # add the label to y
                y_raw.append(class_dict[label])
    # convert the training data lists into the appropriate numpy array shape and format for scikit-learn
    X = np.array(X_raw)
    y = np.array(y_raw)
    print((X.shape, y.shape))
    
   
    
    #X = np.concatenate([X, ndvi, ndwi], axis=1)
    return X,y


folder = 'C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/*_VH.tif'
gl = glob.glob(folder)
print(gl)

#matching = [s for s in xs if "abc" in s]
x_oct, y_oct = feat('C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif',df,month[0])
#print(features('C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif',df))



'''
folder = 'C:/Users/gtvol/Downloads/campo/Dataset/Sentinel-1 (zip)/images/*_VH.tif'
df_feat = extract_features_from_folder(folder)
print(df_feat)
'''