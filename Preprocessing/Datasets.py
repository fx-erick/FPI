"""
Contains different implementations of the torch.utils.data.Dataset class.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image



class SegmentationDataSetNPZ(Dataset):
    """
    Preprocessing class for RGB images and masks.
    Uses pyTorch default normalization values for ImageNet, these work well on most image datasets ...

    :param img_paths: [(str)] python list of strings with paths to images (one .npz file per image)
    :param mask_paths: [(str)] python list of strings with paths to masks (one .npz file per mask)
    :param p_flips: (float) if set, probability with which random horizontal and vertical flips are
                    are applied, defaults to None
    :param p_noise: (float) if set, probability with which random noise is applied, interval (0.0, 1.0]
                    defaults to None
    """

    def __init__(self, img_paths, mask_paths, p_flips=None, p_noise=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.p_flips = p_flips
        self.p_noise = p_noise
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))  # commonly used mean/std for imgs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load .npz to numpy float32 and scale to [0, 1]
        img = np.float32(np.load(self.img_paths[idx])['arr_0']) * 0.00392157
        mask = np.float32(np.load(self.mask_paths[idx])['arr_0']) * 0.00392157

        # change shape to channels first
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        # normalize image
        img = self.norm(img)

        # random flips
        if self.p_flips is not None:
            if torch.rand(1) < self.p_flips:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) < self.p_flips:
                img = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)

        # adding noise to image tensor
        if self.p_noise is not None:
            if torch.rand(1) < self.p_noise:
                # apply noise (roughly from -0.1 to +0.1)
                img = img + torch.randn_like(img) * 0.1

        #print(self.img_paths[idx])
        return img, mask

class SegmentationDataSetPNG(Dataset):
    
    def __init__(self, img_paths, mask_paths, p_flips=None, p_noise=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.p_flips = p_flips
        self.p_noise = p_noise
        self.norm = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))  # commonly used mean/std for imgs

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path =self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = np.float32(Image.open(img_path))* 0.00392157
        mask = np.float32(Image.open(mask_path))* 0.00392157
        if mask.ndim == 3:
            mask = np.mean(mask, axis=2)
        
        mask = mask[...,np.newaxis]
        
        # change shape to channels first
        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)
                    
        # normalize image
        img = self.norm(img)

        # random flips
        if self.p_flips is not None:
            if torch.rand(1) < self.p_flips:
                img = transforms.functional.hflip(img)
                mask = transforms.functional.hflip(mask)
            if torch.rand(1) < self.p_flips:
                img = transforms.functional.vflip(img)
                mask = transforms.functional.vflip(mask)

        # adding noise to image tensor
        if self.p_noise is not None:
            if torch.rand(1) < self.p_noise:
                # apply noise (roughly from -0.1 to +0.1)
                img = img + torch.randn_like(img) * 0.1

        return img, mask

class ClassificationDataSetfromList(Dataset):
    """
    Preprocessing class for 2D data (e.g. images...)
    :param data: list of np.arrays with data
    :param labels: list of np.arrays of shape (1, ) with labels, np.long for classy, np.float32 for regression
    :param transform: torchvision transforms or DatasetTransforms
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
            label = torch.from_numpy(label)
        return data, label


if __name__ == "__main__":
    pass
