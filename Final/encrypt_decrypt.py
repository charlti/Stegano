#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os 
import random
import secrets
import matplotlib.pyplot as plt
from PIL import Image
from preprocess_img import preprocess_image
from preprocess_img import select_random_image


image_folder =  'C:\\Users\\charl\\OneDrive\\Documents\\Projet_SAFE_final\\image_folder'

#Génération de clé 
def generate_random_seed():
    """Génère un seed aléatoire."""
    return random.randint(0, 2**32 - 1)

def generate_key_from_seed(seed, img_size):
    """Génère une clé à partir d'un seed donné, de taille équivalente aux données de l'image."""
    np.random.seed(seed)
    total_size = img_size[0] * img_size[1] 
    return np.random.randint(0, 256, total_size, dtype=np.uint8)

# Chiffrement
def encrypt_image(img, seed):
    """Chiffre une image en niveaux de gris en utilisant une clé générée à partir d'un seed."""
    key = generate_key_from_seed(seed, img.size)
    img_array = np.array(img)
    img_bytes = img_array.flatten()
    key_expanded = key[:len(img_bytes)]
    encrypted_bytes = np.bitwise_xor(img_bytes, key_expanded)
    encrypted_img_array = encrypted_bytes.reshape(img_array.shape)
    encrypted_img = Image.fromarray(encrypted_img_array.astype(np.uint8))
    encrypted_matrix = np.array(encrypted_img)
    return encrypted_img , encrypted_matrix, key

image = select_random_image(image_folder) #Image originale choisie aléatoirement
img_preprocessed = preprocess_image(image)#Image prétraitée et tableau des prédictions

#chiffrement
seed = generate_random_seed() 
encrypted_img, encrypted_img_matrix, key_image = encrypt_image(img_preprocessed, seed)#Image chiffrée 






