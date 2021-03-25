import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# Based on https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/combine_A_and_B.py

##
# This program performs a local dataset combination.
# Path A and path B must exist and contain subfolders named 'train' and 'test'
# A represents the input and B represents the expected output
##

# Paths variables (change these to use your own dataset)
PATH_A =  os.path.realpath("informed_plans/A")          # Represents the input image
PATH_B =  os.path.realpath("informed_plans/B")          # Represents the target image
PATH_AB = os.path.realpath("informed_plans/AB")         # Represents the combined folder

# Function to write the combined images
def write_image(path_A, path_B, path_AB):
    im_A = Image.open(path_A)
    im_B = Image.open(path_B)
    
    im_A = add_border(im_A)
    im_B = add_border(im_B)
    im_AB = combine_imgs(im_A, im_B)
    im_AB.save(path_AB)

def add_border(img):
    ow, oh = img.size
    w = max(ow, oh)
    out = Image.new('RGB', (w, w))
    out.paste(img)
    return out

def combine_imgs(img_a, img_b):
    w, h = img_a.size
    out = Image.new('RGB', (w * 2, h))
    out.paste(img_a)
    out.paste(img_b, (w, 0))
    return out
    

# Get the name of the buckets to be populated (default should be 'test' and 'train')
buckets = os.listdir(PATH_A)

# Iterate through the buckets
for bucket in buckets:
    # Define the folders for input and output images
    img_fold_A = os.path.join(PATH_A, bucket)
    img_fold_B = os.path.join(PATH_B, bucket)

    # Define the folder for the combined images
    img_fold_AB = os.path.join(PATH_AB, bucket)

    # List the images to be combined
    img_list = os.listdir(img_fold_A)
    
    # Create the destination folder if it does not exist
    if not os.path.isdir(img_fold_AB):
        os.makedirs(img_fold_AB)
    
    # Iterate through the images
    img_count = len(img_list)
    for n in tqdm(range(img_count)):

        # Get the name of the image in folder A and its resulting path
        name_A = img_list[n]
        path_A = os.path.join(img_fold_A, name_A)
        
        # Define the name of the image in folder B based on the name in folder B, same for path
        name_B = name_A
        path_B = os.path.join(img_fold_B, name_B)
        
        # Confirm that images have the same name in both folders
        if not os.path.isfile(path_B):
            ext_A = name_A.split('.')[1]
            ext_B = ""
            # If not, try a different file extension (SHOULD BE EITHER png OR jpg)
            if ext_A.lower() == "jpg":
                ext_B = ".png"
            elif ext_A.lower() == "png":
                ext_B = ".jpg"
            
            name_B = name_B.split('.')[0] + ext_B
            path_B = os.path.join(img_fold_B, name_B)
        
        if os.path.isfile(path_A) and os.path.isfile(path_B):
            # Define the name and path of the combined file
            name_AB = name_A
            path_AB = os.path.join(img_fold_AB, name_AB)
            # Write the combined file
            write_image(path_A, path_B, path_AB)