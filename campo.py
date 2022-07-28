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
import os 
import pandas as pd


def offset(length, angle):
    """Retorna o deslocamento em pixels para um determinado comprimento e ângulo"""
    dv = length * np.sign(-np.sin(angle)).astype(np.int32)
    dh = length * np.sign(np.cos(angle)).astype(np.int32)
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
    cooc = np.zeros(shape=shape, dtype=np.int32)
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
  #  print(cooccurrence_maps.shape)
   # print(cooccurrence_maps.shape[2:])
    Nr, Na = cooccurrence_maps.shape[2:]
    glcms = np.zeros(shape=(levels, levels, Nr, Na), dtype=np.float32)
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

def haralick_features(img, win, d, theta, levels, props,file,df, month,class_dict,name_save):
    """Retornar um mapa de caracteristicas de Haralick (um vetor de caracteristicas por pixel)"""
    rows, cols = img.shape
    margin = win + max(d)
    arr = np.pad(img, margin, mode='reflect')
    n_features = len(d) * len(theta) * len(props)
    feature_map = np.zeros(shape=(rows, cols, n_features), dtype=np.float32)
        
    X_raw = []
    y_raw = []
    
    name_save_feat_path = name_save+str('.csv')
    
    #Leitura do raster
    with rasterio.open(file, 'r') as src:
        #Iteração para leitura de cada geometry
        i = 0
        for (label, geom) in tqdm(zip(df[month], df.geometry)):             
            # Le os dados raster correspondentes aos limites da geometria
            window = bounds_window(geom.bounds, src.transform)
            # Armazene nossas informações de Window
            window_affine = src.window_transform(window)
            fsrc = src.read(window=window)
            # Rasteriza  a geometry na forma maior e afina
            mask = rasterize(
                [(geom, 1)],
                out_shape=fsrc.shape[1:],
                transform=window_affine,
                fill=0,
                dtype='float32',
                all_touched=True
            ).astype(bool)
           
            # para cada pixel de label (lugares onde a mask é true)
            label_pixels = np.argwhere(mask)
            # Repetição de cada pixel na geometry
            for (row, col) in label_pixels:
                # Mapa de coorelação de acordo com todos os atributos distancia/windows/angulos/cores
                coocs = cooc_maps(arr, (row + margin, col + margin), win, d, theta, levels)
                # Calcula GLCM
                glcms = compute_glcms(coocs, levels)
                # Calcula as features 
                feat = compute_props(glcms, props) 
                # Adiciona em um vetor os valores de feautures (X) e as classes (Y)
                X_raw.append(feat)
                y_raw.append(class_dict[label])
              
        #Realiza a conversa dos vetores em dataframes e salva em um csv     
        X = np.array(X_raw)
        y = np.array(y_raw)
        df = pd.concat([pd.DataFrame(X),pd.DataFrame(y)],axis = "columns")
        df.to_csv(name_save_feat_path, index=False)
        print("Save data in csv : "+name_save_feat_path)
        return X,y
