import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import os

IMG_EXTENSIONS = [
    '.jpg', '.JPG', 'jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

class ImageFolderLoader():
    def __init__(self, root, phase='train', preprocess='resize_and_crop',
                grayscale=False, method=Image.NEAREST, convert=True, flip=True):
        self.dir_AB = os.path.join(root, phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        if grayscale:
            self.input_nc = 1
            self.output_nc = 1
        else:
            self.input_nc = 3
            self.output_nc = 3
        
        if len(self.dir_AB) == 0:
            raise(RuntimeError("Found 0 images in " + root + "\n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.root = root
        #self.imgs = imgs
        self.transform = get_transform(preprocess=preprocess, grayscale=grayscale,
                                            method=method, convert=convert, flip=flip)
        
        self.loader = default_loader
    
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        A_transform = self.transform
        B_transform = self.transform

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}
    
    def __len__(self):
        return len(self.AB_paths)
    
        
def __scale_width(img, target_size, crop_size, method=Image.NEAREST):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)

def get_transform(preprocess='resize_and_crop', grayscale=False, method=Image.NEAREST, convert=True, flip=True):
        transforms_list = []
        if grayscale:
            transforms_list.append(transforms.Grayscale(1))
        if 'resize' in preprocess:
            osize = [286, 286]
            transforms_list.append(transforms.Resize(osize, method))
        
        elif 'scale_width' in preprocess:
            transforms_list.append(transforms.Lambda(lambda img: __scale_width(img, 
                                    286, 256, method=method)))

        if 'crop' in preprocess:
            transforms_list.append(transforms.RandomCrop(256))
        
        if flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        
        if convert:
            transforms_list += [transforms.ToTensor()]
            if grayscale:
                transforms_list += [transforms.Normalize((0.5,),(0.5,))]
            else:
                transforms_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        
        return transforms.Compose(transforms_list)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    
def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    
    return images[:min(max_dataset_size, len(images))]

def default_loader(path):
    return Image.open(path).convert('RGB')