# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:11:50 2019

@author: Anthony
"""

"""
TUTO

Commencez par exécuter le fichier une fois sans le modifier.
Ensuite, décommentez la ligne 21 et remplacez l'argument "imagepath.png" par
un chemin vers une image sous format PNG que vous souhaitez filtrer.
Enfin, décommentez la ligne   et remplacez, si vous le souhaitez,
noir et blanc par 2 autres couleurs. 
Quelques exemples de couleurs sont dejà définis aux lignes 29 à 37. 
Si vous souhaitez utiliser une autre couleur, il vous suffit d'en créer une
nouvelle en utilisant le code RGB, comme dans les exemples.
"""

#image = imread("imagepath.png")

#filtrer(image, noir, blanc)

from matplotlib.image import *
import maplotlib.pyplot as pl
import numpy as np

blanc = (255, 255, 255)
noir = (0, 0, 0)
rose = (253, 108, 158)
bleu = (44, 117, 255)
vert = (0, 255, 127)
violet = (102, 0, 153)
jaune = (255, 255, 0)
marron = (91, 60, 17)
rouge = (255, 0, 0)

def luminosite(pixel) :
    return sum(pixel) / 3

def mix(pixel, color1, color2) :
    lum = luminosite(pixel)
    return ((color1[0]*(1-lum) + color2[0]*lum)/255, (color1[1]*(1-lum) + color2[1]*lum)/255, (color1[2]*(1-lum) + color2[2]*lum)/255)

def filtrer(image, couleur1, couleur2) :
    figure = pl.figure()
    axes = pl.axes()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)
    shape = image.shape
    for i in range(shape[0]) :
        for j in range(shape[1]) :
            image[i][j] = mix(image[i][j], couleur1, couleur2)
    pl.imshow(image)
    pl.show()







