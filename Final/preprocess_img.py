#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os 
import random
from PIL import Image
import matplotlib.pyplot as plt


image_folder =  'path_to_image_folder'

def select_random_image(image_folder):
    """Sélectionne une image au hasard dans le dossier spécifié et la retourne en niveaux de gris."""
    # Liste tous les fichiers image dans le dossier
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    # Sélectionne une image au hasard
    selected_image_path = os.path.join(image_folder, random.choice(image_files))
    
    # Ouvre l'image et la convertit en niveaux de gris
    img = Image.open(selected_image_path).convert('L')
    
    return img

def preprocess_image(original_image):
    
    # Convertir l'image PIL en tableau numpy
    original_array = np.array(original_image, dtype=np.int32)
    m, n = original_array.shape
    preprocessed_image = np.zeros((m, n), dtype=np.int32) 

    # Initialisation du premier pixel
    preprocessed_image[0, 0] = original_array[0, 0]
    #predictions_array[0, 0] = original_array[0, 0]
    for i in range(m):
        for j in range(n):
            if i == 0 and j == 0:
                continue  # premier pixel déjà initialisé
            elif i == 0:  # Première ligne
                pred_ij = preprocessed_image[i, j-1]  # le pixel à gauche pour la prédiction
            elif j == 0:  # Première colonne
                pred_ij = preprocessed_image[i-1, j]  # le pixel au-dessus pour la prédiction
            else:
                pred_ij = (preprocessed_image[i-1, j] + preprocessed_image[i, j-1]) // 2
                
            #On enregistre le tableau de prédictions    
            #predictions_array[i, j] = pred_ij
            
            pixel_value = original_array[i, j]
            inv_ij = (pixel_value + 128) % 256
            delta = abs(pred_ij - pixel_value)
            delta_inv = abs(pred_ij - inv_ij)

            if delta >= delta_inv:
                if pixel_value < 128:
                    preprocessed_image[i, j] = max(0, pred_ij - 63)
                else:
                    preprocessed_image[i, j] = min(255, pred_ij + 63)
            else:
                preprocessed_image[i, j] = pixel_value
    # Assurer que les valeurs sont dans l'intervalle de uint8 avant de convertir en image PIL
    preprocessed_image = np.clip(preprocessed_image, 0, 255).astype(np.uint8)
    # Convertir le tableau numpy en image PIL
    preprocessed_image_pil = Image.fromarray(preprocessed_image)         
    return preprocessed_image_pil

img = select_random_image(image_folder)
img_preprocessed = preprocess_image(img)



