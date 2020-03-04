import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class ImageNet():
    
    def __call__(self, idx):
        #get pil image and convert to rgb tensors
        
        pil2tensor = transforms.ToTensor()
        tensor2pil = transforms.ToPILImage()
        
        pil=Image.open("/datasets/imagenet-ds/train_64x64/"+idx)
        rgb_image = pil2tensor(pil)
        
        r_image = rgb_image[0]
        g_image = rgb_image[1]
        b_image = rgb_image[2]
        
        #convert to gray scale
        
        grayscale_image = (0.4*r_image + 0.4*g_image + 0.2*b_image).div(3.0)
        
        return(grayscale_image, rgb_image)
        