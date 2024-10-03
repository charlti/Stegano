#!/usr/bin/env python
# coding: utf-8

from encrypt_decrypt import generate_random_seed, encrypt_image, generate_key_from_seed
from preprocess_img import preprocess_image, select_random_image
from cryptography.fernet import Fernet
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ***Prétraitement de l'image*** 


image_folder =  'path_to_image_folder'

image = select_random_image(image_folder)
img_preprocessed = preprocess_image(image)

# Affichage images 
plt.figure(figsize=(10, 5))
# Afficher l'image originale
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')
# Afficher l'image prétraitée
plt.subplot(1, 2, 2)
plt.imshow(img_preprocessed, cmap='gray')
plt.title('Image prétraitée')
plt.axis('off')
plt.show()


# ***Chiffrement de l'image***


seed = generate_random_seed() 
encrypted_img, encrypted_img_matrix, key_image = encrypt_image(img_preprocessed, seed)
#Afficher les images 
plt.figure(figsize=(10, 5))  # Vous pouvez ajuster la taille selon vos besoins

# Image pré-traitée originale
plt.subplot(1, 2, 1)
plt.imshow(img_preprocessed, cmap='gray')  # Remplacez img_preprocessed par img_preprocessed_array si nécessaire
plt.title('Image pré-traitée originale')
plt.axis('off') 

# Image chiffrée
plt.subplot(1, 2, 2)
plt.imshow(encrypted_img, cmap='gray')  # Remplacez encrypted_img par encrypted_img_array si nécessaire
plt.title('Image chiffrée')
plt.axis('off')  # Enlève les axes

# Afficher les images
plt.show()


#ICI, ON TRAVAILLE AVEC ENCRYPTED_MATRIX_VEC 

message_a_chiffrer = 'CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents, bio-informatique… avec des applications notamment dans les secteurs de l’industrie du commerce, des technologies pour la santé, des smart grids.CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents, bio-informatique… avec des applications notamment dans les secteurs de l’industrie du commerce, des technologies pour la santé, des smart grids.CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents.CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents, bio-informatique… avec des applications notamment dans les secteurs de l’industrie du commerce, des technologies pour la santé, des smart grids.CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents, bio-informatique… avec des applications notamment dans les secteurs de l’industrie du commerce, des technologies pour la santé, des smart grids.CRIStAL (Centre de Recherche en Informatique, Signal et Automatique de Lille) est une unité mixte de recherche (UMR 9189) résultant de la fusion du LAGIS (Laboratoire d’Automatique, Génie Informatique et Signal - UMR 8219) et du LIFL (Laboratoire d’Informatique Fondamentale de Lille - UMR 8022) pour fédérer leurs compétences complémentaires en sciences de l’information. CRIStAL est né le 1er janvier 2015 sous la tutelle du CNRS, de l’Université Lille 1 et de l’Ecole Centrale de Lille en partenariat avec l’Université Lille 3, Inria et l’Institut Mines Telecom. CRIStAL est membre de l’institut de recherches interdisciplinaires IRCICA – USR CNRS 3380 (www.ircica.univ-lille1.fr). L’unité est composée de près de 430 membres (222 permanents et plus de 200 non permanents) dont 22 permanents CNRS et 27 permanents Inria. Les activités de recherche de CRIStAL concernent les thématiques liées aux grands enjeux scientifiques et sociétaux du moment tels que : BigData, logiciel, image et ses usages, interactions homme-machine, robotique, commande et supervision de grands systèmes, systèmes embarqués intelligents.'
#Chiffrement du message avec FERNET
def generate_encrypted(message, enc_key):
  
    fernet = Fernet(enc_key)
    encrypted_message = fernet.encrypt(message.encode())

    # Convert the message to bits
    bit_string = bin(int.from_bytes(encrypted_message, 'big'))[2:]
    nb_bits = len(bit_string)
    #print('Taille du message:', nb_bits, 'bits')

    return encrypted_message, bit_string, nb_bits, fernet

#Récupération des MSB de l'image chiffrée 
def get_msb_from_pixels(pixel_values):

    msb_list = []
    for value in pixel_values:
        # Récupère le MSB pour une valeur de pixel donné.
        # Ici, on décale de 7 bits pour un octet (8 bits), ce qui signifie qu'on extrait le bit le plus à gauche.
        msb = (value >> 7) & 1
        msb_list.append(msb)
    
    return np.array(msb_list)

#Modification des MSB de l'image chiffrée par les bits du message chiffré
def modify_msb(msb_list, nb_bits, bit_string):
    msb_list[1:33]= list(np.binary_repr(nb_bits, width=32)) # les 32 premiers bits sont la taille du message
    msb_list[33:33+nb_bits] = list(bit_string) # les nb_bits suivants sont les bits du message  
    return msb_list

#Replacement des MSB modifiés dans l'image chiffrée, on obtient l'image chiffrée marquée (contenant le message chiffré)
def replace_msb(pixel_values, modified_msb):
    modified_pixels = []
    for i in range(len(pixel_values)):
        # Mettre le MSB Ã  0
        new_pixel_value = pixel_values[i] & 127
        # Ajouter le nouveau MSB
        new_pixel_value += modified_msb[i] << 7
        modified_pixels.append(new_pixel_value)
    return np.array(modified_pixels)


encrypted_img_matrix_vec = np.ravel(encrypted_img_matrix) # On transforme notre matrice 2 dimensions en 1 dimensions
msb_values = get_msb_from_pixels(encrypted_img_matrix_vec)


encryption_key = Fernet.generate_key()
encrypted_message, bit_string, nb_bits, key_msg = generate_encrypted(message_a_chiffrer, encryption_key)
#LA CLE DE CHIFFREMENT DU MESSAGE EST STOCKEE DANS LA VARIABLE FERNET
#print("encrypted_message : ", encrypted_message,  "\n")

modified_msb = modify_msb(msb_values, nb_bits, bit_string)

# Utilisation de la fonction pour modifier les valeurs des pixels
modified_pixel_values = replace_msb(encrypted_img_matrix_vec, modified_msb)

# Transformation des valeurs des pixels modifiÃ©es en une matrice ayant la mÃªme forme que l'image originale
modified_img_matrix = np.array(modified_pixel_values).reshape(encrypted_img_matrix.shape)
modified_img = modified_img_matrix
modified_img_matrix = np.ravel(modified_img_matrix)
print("nb_bits_message_chiffre : ", nb_bits,"\n")
print("message à chiffrer :", message_a_chiffrer,"\n")

# Conversion de la matrice modifiÃ©e en une image
encrypted_img_marked = Image.fromarray(modified_img.astype(np.uint8))
# Utilisation de plt.imshow pour afficher l'image directement à partir de la matrice NumPy
plt.imshow(encrypted_img_marked, cmap='gray')  # 'cmap=gray' est pour les images en niveaux de gris. Enlevez-le pour les images en couleur.
plt.title('Image chiffrée marquée ')
plt.axis('off')  # Enlève les axes pour une visualisation claire
plt.show()


# ***Récupération du message depuis l'image chiffrée***

# ***1)Récupération_taille_message_inséré***

#Récupérer la taille du message inséré dans l'image, on sait que la taille est indiquée du 1er au 33ème pixel
def get_msb_from_pixels_modified(pixel_values):
    msb_list = []
    # Assurez-vous que la liste des valeurs de pixels est assez grande
    if len(pixel_values) >= 33:
        # Récupère les MSB des pixels d'indice 1 Ã  32 (33 non inclus)
        for value in pixel_values[1:33]:
            # Récupère le MSB pour une valeur de pixel donnÃ©e.
            msb = (value >> 7) & 1
            msb_list.append(msb)
    else:
        raise ValueError("La liste des pixels n'est pas assez grande pour extraire les MSB demandÃ©s.")

    return np.array(msb_list)

# Taille du message inséré
message_length_decimal = int(''.join(str(bit) for bit in get_msb_from_pixels_modified(modified_img_matrix)), 2)
print("\033[1m\033[91mnb_bits_msg_insere:\033[0m", message_length_decimal, "bits")


# ***2) Récupération du message***

modified_img_matrix_new = modified_img_matrix[33:33+nb_bits]
message_insere = get_msb_from_pixels(modified_img_matrix_new)

# ***3) Décodage du message***

def generate_decrypted(message, key):
    bit_str = '0b'
    for i in range(len(message)):
        if message[i]==0:
            bit_str=bit_str+'0'
        else:
            bit_str=bit_str+'1'

    n = int(bit_str, 2)
    decoded_message = key.decrypt(n.to_bytes((n.bit_length() + 7) // 8, 'big')).decode()


    return decoded_message
#Utilisation de la clé FERNET
decoded_message=generate_decrypted(message_insere, key_msg)
print("\033[1m\033[91mmessage récupéré de l'image :\033[0m", decoded_message)


# ***4) Déchiffrement de l'image***

#Déchiffrement image
def partially_decrypt_image(encrypted_img, key):
    """Déchiffre une image en niveaux de gris en utilisant la clé fournie."""
    # Assurez-vous que la clé a la bonne taille pour effectuer l'opération XOR avec l'image chiffrée
    img_array = np.array(encrypted_img)
    img_bytes = img_array.flatten()
    key_expanded = key[:len(img_bytes)]
    
    # Effectuer l'opération XOR avec la clé pour déchiffrer les octets de l'image
    partially_decrypted_bytes = np.bitwise_xor(img_bytes, key_expanded)
    partially_decrypted_img_array = partially_decrypted_bytes.reshape(img_array.shape)
    
    # Convertir l'array déchiffré en image PIL pour le retourner
    partially_decrypted_img = Image.fromarray(partially_decrypted_img_array.astype(np.uint8))
    return partially_decrypted_img, partially_decrypted_img_array  

partially_decrypted_img, partially_decrypted_img_array = partially_decrypt_image(encrypted_img_marked, key_image)

plt.figure(figsize=(15, 5))
plt.imshow(partially_decrypted_img, cmap='gray')  # Assurez-vous que l'affichage est correct pour les images décryptées
plt.title('Image partiellement déchiffrée')
plt.axis('off')
plt.show()


# ***Prédiction des pixels sur l'image partiellement déchiffrée, Obtention des pred(i,j)***

def calculate_pred(decrypted_img_array, i, j):
    if i == 0 and j == 0:
        # Le premier pixel n'a pas été modifié
        return decrypted_img_array[0, 0]
    if i == 0:
        # Première ligne, on prend le pixel à gauche
        return decrypted_img_array[i, j-1]
    if j == 0:
        # Première colonne, on prend le pixel au-dessus
        return decrypted_img_array[i-1, j]
    # Pour les autres cas, prendre la moyenne des pixels au-dessus et à gauche
    return (decrypted_img_array[i-1, j] + decrypted_img_array[i, j-1]) // 2


# ***Prédiction des MSB et Reconstruction de l'image*** 


def reconstruct_image_with_predicted_msb(partially_decrypted_image):
    partially_decrypted_img_array = np.array(partially_decrypted_image, dtype=np.int32)
    m, n = partially_decrypted_img_array.shape
    fully_reconstructed_img_array = np.zeros((m, n), dtype=np.uint32)

    for i in range(m):
        for j in range(n):
            # Calculer les valeurs de pixel avec MSB à 0 et MSB à 1, donc deux versions hypothétiques de chaque pixel (MSB 0 et MSB 1)
            pixel_value_with_MSB_0 = partially_decrypted_img_array[i, j] & 0x7F # Mettre le MSB à 0 0x7F masque héxa de 01111111
            pixel_value_with_MSB_1 = partially_decrypted_img_array[i, j] | 0x80 # Mettre le MSB à 1 0x80 masque héxa de 10000000
            #Calcul des prédictions 
            pred= calculate_pred(fully_reconstructed_img_array,i,j)
            # Calculer les différences Δ⁰ et Δ¹
            delta_0 = abs(pred - pixel_value_with_MSB_0)
            delta_1 = abs(pred - pixel_value_with_MSB_1)

            # Déterminer le MSB correct
            correct_MSB = 0 if delta_0 <= delta_1 else 1

            # Reconstruire la valeur complète du pixel
            fully_reconstructed_img_array[i, j] = pixel_value_with_MSB_0 if correct_MSB == 0 else pixel_value_with_MSB_1
            
    # Assurer que les valeurs sont dans l'intervalle de uint8 avant de convertir en image PIL
    fully_reconstructed_img_array = np.clip(fully_reconstructed_img_array, 0, 255).astype(np.uint8)
    # Convertir le tableau numpy en image PIL
    fully_reconstructed_img = Image.fromarray(fully_reconstructed_img_array) 

    return fully_reconstructed_img

fully_reconstructed_img = reconstruct_image_with_predicted_msb(partially_decrypted_img)

#Affichage des images 
plt.figure(figsize=(10, 5))
# Afficher l'image originale prétraitée
plt.subplot(1, 2, 1)
plt.imshow(img_preprocessed, cmap='gray')
plt.title('Image originale prétraitée')
plt.axis('off')
# Afficher l'image reconstruite
plt.subplot(1, 2, 2)
plt.imshow(fully_reconstructed_img, cmap='gray')
plt.title('Image reconstruite')
plt.axis('off')
plt.show()

#################################################### ANALYSE DES METRIQUES #################################################################################

"""
# ***Déclaration des tableaux des métriques***

#tableau de valeurs pour chaque métrique
psnr_values = []
ssim_values = []
corr_horizontal_values = []
corr_vertical_values = []
#entropy_values_encrypted = []
#entropy_values_marked_encrypted = []
#payload_values  = []
npcr_values  = []
#uaci_values  = []


# ***Evaluation qualité des résultats METRICS*** 

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

#PSNR
def calculate_psnr(original_image, reconstructed_image):
    
    # Convert images to numpy arrays
    original_np = np.array(original_image)
    reconstructed_np = np.array(reconstructed_image)
    
    # Validate that both images have the same dimensions
    if original_np.shape != reconstructed_np.shape:
        raise ValueError("Images must have the same dimensions for PSNR calculation.")
    
    # Calculate MSE (Mean Squared Error)
    mse = np.mean((original_np - reconstructed_np) ** 2)
    if mse == 0:
        # Means the two images are identical; PSNR has no meaning.
        return float('inf')
    
    # Calculate max pixel value (assuming 8-bit depth)
    max_pixel = 255.0
    
    # Calculate PSNR
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr_value

#SSIM
def calculate_ssim(original_img, processed_img):
    Calcule le SSIM entre l'image originale et l'image déchiffrée. SANS UNITE
    original_array = np.array(original_img)
    processed_array = np.array(processed_img)
    return ssim(original_array, processed_array, data_range=255)

#Corrélation
def correlation_coefficients(image):
    # Convert image to a numpy array if it isn't already
    image = np.array(image)

    # Calculate the means
    mean_image = np.mean(image)
    
    # Horizontal correlation
    # Difference between adjacent horizontal pixels
    horizontal_diff = image[:, :-1] - image[:, 1:]
    # Calculate the mean of the differences
    mean_horizontal_diff = np.mean(horizontal_diff)
    # Calculate the sample variance of the differences
    var_horizontal_diff = np.var(horizontal_diff)
    # Correlation coefficient for horizontal adjacent pixels
    corr_horizontal = np.mean((horizontal_diff - mean_horizontal_diff)**2) / var_horizontal_diff
    
    # Vertical correlation
    # Difference between adjacent vertical pixels
    vertical_diff = image[:-1, :] - image[1:, :]
    # Calculate the mean of the differences
    mean_vertical_diff = np.mean(vertical_diff)
    # Calculate the sample variance of the differences
    var_vertical_diff = np.var(vertical_diff)
    # Correlation coefficient for vertical adjacent pixels
    corr_vertical = np.mean((vertical_diff - mean_vertical_diff)**2) / var_vertical_diff
    
    return corr_horizontal, corr_vertical

#Calcul d'entropie de Shannon pour l'image chiffrée/image chiffrée marquée pour évaluer le caractère uniforme de l'image
def shannon_entropy_image(image):
    
    # Load image and convert to grayscale
    image = np.array(image)

    # Flatten the image to 1D array for histogram computation
    pixel_counts = np.bincount(image.ravel(), minlength=256)

    # Normalize the histogram to get the probabilities
    total_pixels = image.size
    probabilities = pixel_counts / total_pixels

    # Compute Shannon entropy
    entropy = -np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0]))
    
    return entropy


#Calcul NPCR 
def calculate_npcr(image1, image2):
    
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for NPCR calculation.")
    
    # Calculate d(i, j) for all pixels
    d_matrix = image1 != image2
    npcr = np.sum(d_matrix) / d_matrix.size * 100
    return npcr

#Calcul UACI
def calculate_uaci(image1, image2):
    # Convert images to numpy arrays if they aren't already
    image1 = np.array(image1)
    image2 = np.array(image2)
    
    # Check if both images have the same dimensions
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same dimensions for UACI calculation.")
    
    # Calculate the absolute intensity difference between the images
    intensity_diff = np.abs(image1 - image2)
    
    # Calculate UACI according to the provided formula
    uaci = 100 * np.sum(intensity_diff) / (image1.size * 255)
    
    return uaci

psnr_values.append(calculate_psnr(image,img_preprocessed))
ssim_values.append(calculate_ssim(image,img_preprocessed))
corr_horizontal, corr_vertical = correlation_coefficients(image)
corr_horizontal_values.append(corr_horizontal)
corr_vertical_values.append(corr_vertical)
#entropy_values_encrypted.append(shannon_entropy_image(encrypted_img))
#entropy_values_marked_encrypted.append(shannon_entropy_image(encrypted_img_marked))
#payload_values.append(calculate_payload_bpp(encrypted_img, encrypted_img_marked))
npcr_values.append(calculate_npcr(img_preprocessed, encrypted_img_marked))
#uaci_values.append(calculate_uaci(img_preprocessed, encrypted_img_marked))

# Affichage des résultats
print("PSNR Values:", psnr_values)
print("SSIM Values:", ssim_values)
print("Horizontal Correlation Values:", corr_horizontal_values)
print("Vertical Correlation Values:", corr_vertical_values)
#print("Entropy Values Encrypted:", entropy_values_encrypted)
#print("Entropy Values Encrypted Marked:", entropy_values_marked_encrypted)
#print("Payload values :", payload_values)
print ("NPCR values : ", npcr_values)
#print("UACI values : ", uaci_values)


# ***Représentation graphique PSNR***

plt.figure(figsize=(8, 6))
plt.plot(psnr_values, label='PSNR', marker='o', linestyle='-', color='blue')
plt.title('PSNR entre Image Originale et Image Prétraitée')
plt.xlabel('Indice image')
plt.ylabel('PSNR (dB)')
plt.legend()
plt.grid(True)
plt.show()


# ***Représentation graphique SSIM***

# Plot SSIM values
plt.figure(figsize=(8, 6))
plt.plot(ssim_values, label='SSIM', marker='s', linestyle='-', color='red', linewidth=2)
plt.title('SSIM entre image originale et image prétraitée ')
plt.ylabel('Valeur SSIM')
plt.xlabel('Indice image')
plt.legend()
plt.grid(True)
plt.show()


# ***Représentation graphique correlation verticale horizontale image originale***

corr_horizontal_values = [0.92, 0.95, 0.965, 0.92, 0.94, 0.97, 0.975, 0.965]
corr_vertical_values = [0.90, 0.93, 0.945, 0.91, 0.945, 0.96, 0.965, 0.96]
plt.figure(figsize=(8, 6))
plt.plot(corr_horizontal_values, label='Corrélation Horizontale', marker='^', linestyle='--', color='green', linewidth=2)
plt.plot(corr_vertical_values, label='Corrélation Verticale', marker='v', linestyle='-.', color='purple', linewidth=2)
plt.title('Corrélations Horizontales et Verticales des images originales')
plt.ylabel('Valeur de Corrélation')
plt.xlabel('Indice image')
plt.legend()
plt.grid(True)
plt.show()


# ***Représentation graphique entropie Shannon image chiffrée et image chiffrée marquée***


plt.figure(figsize=(8, 6))
plt.plot(entropy_values_encrypted, label='Entropie Images Chiffrées', marker='o', linestyle='-', color='blue')
plt.plot(entropy_values_marked_encrypted, label='Entropie Images Chiffrées Marquées', marker='s', linestyle='-', color='red')
plt.title('Entropie de Shannon des Images Chiffrées et Marquées')
plt.xlabel('Indice de l\'Image')
plt.ylabel('Entropie de Shannon')
plt.legend()
plt.grid(True)
plt.show()


# ***Représentation graphique métrique NPCR entre image prétraitée et image chiffrée marquée***


plt.figure(figsize=(8, 6))
plt.plot(npcr_values, label='NPCR', marker='^', linestyle='-', color='green')
plt.title('Métrique NPCR entre Images prétraitées et Images Chiffrées Marquées')
plt.xlabel('Indice de l\'Image')
plt.ylabel('Valeur NPCR (%)')
 # Set the y-axis to range from 0 to 100
plt.legend()
plt.grid(True)
plt.show()


# ***Représentation graphique métrique UACI entre image prétraitée et image chiffrée marquée***

uaci_values = [30.43601688, 30.64398671, 30.50345422, 30.43110398, 30.2795685, 30.55736764, 30.29698401, 30.86471625]
plt.figure(figsize=(8, 6))
plt.plot(uaci_values, label='UACI', marker='v', linestyle='-', color='purple')
plt.title('Métrique UACI entre Images prétraitées et Images Chiffrées Marquées')
plt.xlabel('Indice de l\'Image')
plt.ylabel('Valeur UACI (%)')  # Définir l'échelle de l'axe des ordonnées de 0 à 100
plt.legend()
plt.grid(True)
plt.show()



"""

