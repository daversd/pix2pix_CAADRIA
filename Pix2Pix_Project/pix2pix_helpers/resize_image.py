from PIL import Image
from PIL import ImageOps
import os


FOLDER = os.path.realpath("informed_plans")




def scale_up():
    """This method scales up the images in the folder by 4
    (2,2) and then crops to the original size, expected to be 256x256
    """
    for (dirpath, dirnames, filenames) in os.walk(FOLDER):
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

def scale_down():
    """This method scales up the images in the folder by 4
    (2,2) and then crops to the original size, expected to be 256x256
    """
    for (dirpath, dirnames, filenames) in os.walk(FOLDER):
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

scale_up()
#scale_down()