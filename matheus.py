# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:49:29 2022

@author: gtvol
"""

import campo
from skimage import io
import numpy as np
from skimage.io import imsave, imread
from skimage.color import rgb2gray
import geopandas as gpd

name_save = '04Dec_2015_VH'
file = 'D:/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif'

raster = rgb2gray(imread(file))
raster = np.array(raster,dtype = np.uint8)
#utilizado para calcular a margem e o número de features tambem para voltar o mapa de coocorencia
d = (1,2)

theta = (0, np.pi/4, np.pi/2, 3*np.pi/4)

props = ('correlation','homogeneity','contrast','dissimilarity')

levels = 256

win = 3

df = gpd.read_file('D:/campo/Dataset/Reference/CampoVerde_Oct2015_Jul2016.shp')


month = ['Oct_2015','Nov_2015','Dec_2015','Jan_2016','Feb_2016','Mar_2016','Apr_2016','May_2016','Jun_2016','Jul_2016']
classes = np.unique(df[month])
#print(classes)
class_dict = dict(zip(classes, range(len(classes))))

Xs, ys = campo.haralick_features(raster, win, d, theta, levels, props,file,df, month[0],class_dict,name_save)
