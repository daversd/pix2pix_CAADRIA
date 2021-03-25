import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
    """
    input_image = input_image.cpu()
    input_image = torch.add(input_image, 1)
    input_image = torch.div(input_image, 2)
    return input_image.permute(1, 2, 0)
    #return input_image

def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)

def save_visuals(visuals, test_dir):
    input = visuals['real_A'].view(3, 256, 256)
    output = visuals['fake_B'].view(3, 256, 256)
    real = visuals['real_B'].view(3, 256, 256)

    input = tensor2im(input).detach().numpy() * 255
    output = tensor2im(output).detach().numpy() * 255
    real = tensor2im(real).detach().numpy() * 255

    out_im = Image.new('RGB', (256 * 3, 256))
    out_im.paste(Image.fromarray(np.uint8(input)))
    out_im.paste(Image.fromarray(np.uint8(output)), (256, 0))
    out_im.paste(Image.fromarray(np.uint8(real)), (512, 0))

    out_im.save(test_dir)

def save_generated(visuals, test_dir):
    #input = visuals['real_A'].view(3, 256, 256)
    output = visuals['fake_B'].view(3, 256, 256)
    #real = visuals['real_B'].view(3, 256, 256)

    #input = tensor2im(input).detach().numpy() * 255
    output = tensor2im(output).detach().numpy() * 255
    #real = tensor2im(real).detach().numpy() * 255

    out_im = Image.new('RGB', (256 , 256))
    #out_im.paste(Image.fromarray(np.uint8(input)))
    out_im.paste(Image.fromarray(np.uint8(output)))
    #out_im.paste(Image.fromarray(np.uint8(real)), (512, 0))
    out_im.save(test_dir)

def plot_visuals(visuals):
    input = visuals['real_A'].view(3, 256, 256)
    output = visuals['fake_B'].view(3, 256, 256)
    real = visuals['real_B'].view(3, 256, 256)
    
    input = tensor2im(input)
    output = tensor2im(output)
    real = tensor2im(real)
    
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(input.detach().numpy())
    axs[0].set_title('Input image')

    
    axs[1].imshow(output.detach().numpy())
    axs[1].set_title('Generated image')

    axs[2].imshow(real.detach().numpy())
    axs[2].set_title('Real image')

    plt.show()