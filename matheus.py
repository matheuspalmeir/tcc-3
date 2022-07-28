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
import time


begin_timer = time.time()   # Começa o a contar o tempo

#TROCAR NOME DE SALVAR, O ARQUIVO QUE VAI LER E TAMBEM O NUMERO DO MES/ANO
name_save = '04Dec_2015_VH'
file = 'D:/campo/Dataset/Sentinel-1 (zip)/images/04Dec_2015_VH.tif'
#Leitura da imagem 
raster = rgb2gray(imread(file))
raster = np.array(raster,dtype = np.float32)

#Distancia dos pixels
d = [1]
#Angulos
theta = (0, np.pi/4, np.pi/2, 3*np.pi/4)
#Propriedades
props = ('correlation','homogeneity','contrast','dissimilarity')
#Tons de cinza
levels = 256
#Tamanho da janela
win = 3
#Leitura do dataframe de shapely para pegar os pixels certos de cada geometry
df = gpd.read_file('D:/campo/Dataset/Reference/CampoVerde_Oct2015_Jul2016.shp')

#Pegar o mes referente de cada fotografia - só realizar a mudança na linha 46 coo mes correto
#            0          1           2         3          4           5          6          7         8         9
month = ['Oct_2015','Nov_2015','Dec_2015','Jan_2016','Feb_2016','Mar_2016','Apr_2016','May_2016','Jun_2016','Jul_2016']
classes = np.unique(df[month])
class_dict = dict(zip(classes, range(len(classes))))

#Passar todos os parametros, só
Xs, ys = campo.haralick_features(raster, win, d, theta, levels, props,file,df, month[0],class_dict,name_save)

end_timer = time.time() # Termina a contagem de tempo do algoritmo
print(end_timer-begin_timer)