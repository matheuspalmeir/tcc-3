# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 17:39:49 2022

@author: gtvol
"""

import numpy as np
from skimage import io
from scipy import stats
from skimage.feature import greycoprops
import time
from tqdm import tqdm
import rasterio
import rasterstats
from rasterio.features import rasterize
from rasterstats.io import bounds_window

def offset(length, angle):
    """Retorna o deslocamento em pixels para um determinado comprimento e ângulo"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int16)
    dh = length * np.sign(np.cos(angle)).astype(np.int16)
    return dv, dh

def crop(img, center, win):
    """Retorne um corte quadrado de imagem centralizado no centro (side = 2*win + 1)"""
    row, col = center
    side = 2*win + 1
    first_row = row - win
    first_col = col - win
    last_row = first_row + side    
    last_col = first_col + side
    return img[first_row: last_row, first_col: last_col]

def cooc_maps(img, center, win, d=[1], theta=[0], levels=256):
    """
    Retorne um conjunto de mapas de coocorrência para diferentes d e teta 
    em um corte quadrado centrado no centro (size = 2*w + 1)
    """
    shape = (2*win + 1, 2*win + 1, len(d), len(theta))
    cooc = np.zeros(shape=shape, dtype=np.int16)
    row, col = center
    Ii = crop(img, (row, col), win)
    for d_index, length in enumerate(d):
        for a_index, angle in enumerate(theta):
            dv, dh = offset(length, angle)
            Ij = crop(img, center=(row + dv, col + dh), win=win)
            cooc[:, :, d_index, a_index] = encode_cooccurrence(Ii, Ij, levels)
    return cooc

def encode_cooccurrence(x, y, levels=256):
    """Retorna o código correspondente à coocorrência das intensidades x e y"""
    return x*levels + y

def decode_cooccurrence(code, levels=256):
    """Retorna as intensidades x, y correspondentes ao código"""
    return code//levels, np.mod(code, levels)    

def compute_glcms(cooccurrence_maps, levels=256):
    """Calcular as frequências de coocorrência dos mapas de coocorrência"""
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float16)
    for r in range(Nr):
        for a in range(Na):
            table = stats.itemfreq(cooccurrence_maps[:, :, r, a])
            codes = table[:, 0]
            freqs = table[:, 1]/float(table[:, 1].sum())
            i, j = decode_cooccurrence(codes, levels=levels)
            glcms[i, j, r, a] = freqs
    return glcms

def compute_props(glcms, props=('contrast',)):
    """Retorna um vetor de caracteristicas correspondente a um conjunto de GLCM"""
    Nr, Na = glcms.shape[2:]
    features = np.zeros(shape=(Nr, Na, len(props)))
   
    for index, prop_name in enumerate(props):
        features[:, :, index] = greycoprops(glcms, prop_name)
    return features.ravel() 

def feat(img,df, month,class_dict):
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
    
    
def haralick_features(img, win, d, theta, levels, props,file,df, month,class_dict):
    """Retornar um mapa de caracteristicas de Haralick (um vetor de caracteristicas por pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float16)
    
    
    # this larger cell reads data from a raster file for each training vector
    X_raw = []
    y_raw = []
    i = 0
    with rasterio.open(file, 'r') as src:
        for (label, geom) in tqdm(zip(df[month], df.geometry)):
           # print(i)
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
                start_time = time.time()
                coocs = cooc_maps(arr, (row + margin, col + margin), win, d, theta, levels)
               # print("---1Tempo cooc_maps %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                glcms = compute_glcms(coocs, levels)
               # print(glcms.shape)
               # print("---2Tempo GLCM %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
                feature_map[row, col, :] = compute_props(glcms, props) 
               # print("---2Tempo feature_map  %s seconds ---" % (time.time() - start_time))
    
                time.sleep(0.0001)
                # add a pixel of data to X
                """
                data = fsrc[:,row,col]
                one_x = np.nan_to_num(data, nan=1e-3)
                X_raw.append(one_x)
                # add the label to y
                y_raw.append(class_dict[label])"""
            time.sleep(0.01)
    # convert the training data lists into the appropriate numpy array shape and format for scikit-learn
    X = np.array(X_raw)
    y = np.array(y_raw)
    print((X.shape, y.shape))
    return feature_map
"""
    for m in range(rows):        
        for n in range(cols):
            start_time = time.time()
            coocs = cooc_maps(arr, (m + margin, n + margin), win, d, theta, levels)
            print("---1Tempo cooc_maps %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            glcms = compute_glcms(coocs, levels)
           # print(glcms.shape)
            print("---2Tempo GLCM %s seconds ---" % (time.time() - start_time))
            start_time = time.time()
            feature_map[m, n, :] = compute_props(glcms, props) 
            print("---2Tempo feature_map  %s seconds ---" % (time.time() - start_time))

            time.sleep(0.0001)
           # print(feature_map.shape)
    return feature_map"""