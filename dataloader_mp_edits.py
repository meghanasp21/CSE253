import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os

class ImageDataset():
    
    
    def __init__(self, path_file, n_samples=None, random_seed=None, transform=None):
        """
        Args:
        path_file : (string)
            file path to a csv file listing all images
        n_samples : (int), optional (default=None)
            number of samples to take out of file path, randomly sampled if specified
            If None, uses all samples (in order from path file)
        random_seed (int), optional
            Used to set random state for reproducable subsampling
            If None, no random state set
        transform (list, optional)
            Optional transform to be applied on a sample, should be a list of torchvision,.
        """
        if n_samples is None:
            self.data = pd.read_csv(path_file)
        else:
            full_data = pd.read_csv(path_file)
            if random_seed is None:
                self.data = full_data.sample(n_samples)
            else:
                self.data = full_data.sample(n_samples, random_state=random_seed)
        self.transform = transform
    
    
    def __getitem__(self, idx):
        """
        Args:
        idx : (int)
            the idx of self.data to grab
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data.iloc[idx, 0]
        
        rgb_init = Image.open(img_name)
        gray_init = Image.open(img_name)
        
        rgb_image = rgb_init.copy()    
        gray_image = gray_init.copy()
             
        if self.transform is not None:
            rgb_transforms = transforms.Compose(self.transform)
            rgb_trans_image = rgb_transforms(rgb_image)

            gray_transform = transforms.Grayscale(num_output_channels=1)
            gray_trans_image = rgb_transforms(gray_image)
            gray_trans_image = gray_transform(gray_trans_image)

        else:
            rgb_trans_image = rgb_image
            gray_transforms = transforms.Grayscale(num_output_channels=1)
            gray_trans_image = gray_transforms(gray_image)
        
        pil2tensor = transforms.ToTensor()
        rgb_tensor = pil2tensor(rgb_trans_image)
        gray_tensor = pil2tensor(gray_trans_image)
        
        return gray_tensor, rgb_tensor
    
    
    def __len__(self):
        return len(self.data)