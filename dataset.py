from torch.utils.data import Dataset
import tifffile
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2

class CellDataset(Dataset):
    def __init__(self, root_dir, border_core=False, split="train", fold=None, transform=None):

        self.transform = transform
        self.border_core = border_core

        root_dir = Path(root_dir)
        if split == "train":
            if fold == "a":
                self.img_files = sorted(list(root_dir.glob(r'[bc]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[bc]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[bc]'+'_meta/*.json')))
            elif fold == "b":
                self.img_files = sorted(list(root_dir.glob(r'[ac]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[ac]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[ac]'+'_meta/*.json')))
            else:
                self.img_files = sorted(list(root_dir.glob(r'[ab]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[ab]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[ab]'+'_meta/*.json')))
        elif split == "val":
            if fold == "a":
                self.img_files = sorted(list(root_dir.glob(r'[a]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[a]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[a]'+'_meta/*.json')))
            elif fold == "b":
                self.img_files = sorted(list(root_dir.glob(r'[b]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[b]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[b]'+'_meta/*.json')))
            else:
                self.img_files = sorted(list(root_dir.glob(r'[c]'+"/*.tif")))
                self.mask_files = sorted(list(root_dir.glob(r'[c]'+"_GT/*.tif"))) 
                self.meta_files = sorted(list(root_dir.glob(r'[c]'+'_meta/*.json')))
        elif split == "test":
            self.img_files = sorted(list(root_dir.glob(r'[de]'+"/*.tif")))
            self.mask_files = None

    def __getitem__(self, idx):
        img = tifffile.imread(self.img_files[idx])
        mask = tifffile.imread(self.mask_files[idx]).astype(np.float32) if self.mask_files else None

        orig_size = img.shape
        file_name = '/'.join(str(self.img_files[idx]).split('/')[-2:])
          
        if self.transform is not None:
            if self.mask_files:
                transformed = self.transform(image=img, mask=mask)
            else:
                transformed = self.transform(image=img)
            img = transformed['image']
            if self.mask_files:
                mask = transformed['mask'].long()
        
        img = img.half()

        if self.mask_files == None:
            return img, orig_size, file_name
        
        return img, mask, orig_size, file_name

    def __len__(self):
        return len(self.img_files)
    

def train_transform():
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.GaussNoise(p=0.2),
            ToTensorV2()
        ])
    
    return transform

def val_transform():
    transform = A.Compose([
            ToTensorV2()
        ])
    
    return transform

def test_transform():
    transform = A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_LINEAR), 
            ToTensorV2()
        ])
    
    return transform