"""Pix2Pix implementation based on 
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Developed by David Dória https://github.com/daversd for
2021 CAADRIA Workshop 1

"""

import torch
import torch.onnx
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pix2pix_helpers.util as util
from pix2pix_helpers.create_dataset import ImageFolderLoader
from pix2pix_helpers.pix2pix_model import Pix2PixModel
from matplotlib import pyplot as plt
import time
import os
import glob
from PIL import Image
from PIL import ImageOps

##
# High level setup
##

#FOLDER_NAME = 'informed_plans'                                   # The name of the data folder
MODEL_NAME = 'informed_run_2'                              # The name of the model for this run
LOAD_NUMBER = -1                                        # Number of the model to be loaded (-1 loads the latest)
CKPT_DIR = os.path.join('checkpoints', MODEL_NAME)      # The folder to save checkpoints to
SAVE_DIR = 'generate'

EPOCHS = 200
BATCH_SIZE = 1

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##
# Main program
##

def load_model(model):
    """
    Loads the networks from the checkpoint specified in LOAD_NUMBER
    Use -1 to load the latest model.
    """
    
    list_of_files = glob.glob(CKPT_DIR + '/*.pth')

    if LOAD_NUMBER == -1:
        file_path = max(list_of_files, key=os.path.getctime)
        file_name = os.path.basename(file_path)
        file_number = file_name.split('_')[0]
        print(file_number)
    else:
        file_number = LOAD_NUMBER
    
    file_prefix = os.path.join(CKPT_DIR, str(file_number) + '_')
    netG_File = file_prefix + 'net_G.pth'
    netD_File = file_prefix + 'net_D.pth'
    
    files_exist = os.path.exists(netG_File) and os.path.exists(netD_File)
    assert files_exist, f"Checkpoint {LOAD_NUMBER} does not exist. Check '{CKPT_DIR}' to see available checkpoints"
    print(f"Loading model from checkpoint {file_number} \n"+ f"Generator is {netG_File} \n" + f"Discriminator is {netD_File}")

    model.load_networks(file_number)

def scale_up(folder):
    """This method scales up the images in the folder by 4
    (2,2) and then crops to the original size, expected to be 256x256
    """
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if '.jpg' in filename:
                filepath = os.path.join(dirpath, filename)
                im = Image.open(filepath)
                (width, height) = im.size

                width = width * 2
                height = height * 2

                left = int(width / 4)
                top = int(height / 4)
                right = left + 256
                bottom = top + 256

                im = im.resize((width, height), resample=Image.NEAREST)
                im = im.crop((left, top, right, bottom))

                im.save(filepath)

def scale_down(folder):
    """This method scales up the images in the folder by 4
    (2,2) and then crops to the original size, expected to be 256x256
    """
    for (dirpath, dirnames, filenames) in os.walk(folder):
        for filename in filenames:
            if '.jpg' in filename:
                filepath = os.path.join(dirpath, filename)
                im = Image.open(filepath)
                (width, height) = im.size

                width = int(width / 2)
                height = int(height / 2)


                im = im.resize((width, height), resample=Image.NEAREST)
                im = ImageOps.expand(im, 64, fill=(255, 255,255))

                im.save(filepath)

if __name__ == '__main__':

    # Create the testing data set
    #testData = ImageFolderLoader(os.path.join(SAVE_DIR, 'source'), phase='test', flip=False, preprocess='none')
    #testSet = torch.utils.data.DataLoader(testData, batch_size=BATCH_SIZE, shuffle= False, num_workers=0)

    # Create the pix2pix model in testing mode
    model = Pix2PixModel(CKPT_DIR, MODEL_NAME, is_train=False, n_epochs=EPOCHS/2, n_epochs_decay=EPOCHS/2)
    model.setup()
    model.eval()
    load_model(model)

    
    data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    i=0
    # Iterate through test data set, for the lenght of the test sample
    scale_up(f"{SAVE_DIR}/source")
    for (dirpath, dirnames, filenames) in os.walk(f"{SAVE_DIR}/source"):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            im = Image.open(filepath).convert('RGB')
            im = data_transforms(im)
            im = im.view(1, 3, 256, 256)
            model.set_input(im, single=True)
            model.test()
            
            visuals = model.get_current_visuals()
            save_path = os.path.join(f'{SAVE_DIR}/result', filename)
            util.save_generated(visuals, save_path)
            i += 1
    scale_down(f"{SAVE_DIR}/source")
    scale_down(f"{SAVE_DIR}/result")