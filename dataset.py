from torch.utils.data import Dataset
import tifffile
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class CellDataset(Dataset):
    def __init__(self, root_dir, border_core=False, split="train", transform=None):

        self.transform = transform
        self.border_core = border_core

        root_dir = Path(root_dir)
        if split == "train":
            self.img_files = sorted(list(root_dir.glob(r'[ab]'+"/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r'[ab]'+"_GT/*.tif"))) 
        elif split == "val":
            self.img_files = sorted(list(root_dir.glob(r'[c]'+"/*.tif")))
            self.mask_files = sorted(list(root_dir.glob(r'[c]'+"_GT/*.tif"))) 
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
        img = np.tile(img, (3,1,1))

        return img, mask, orig_size, file_name

    def __len__(self):
        return len(self.img_files)
    

def train_transform():
    transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            ToTensorV2()
        ])
    
    return transform

def val_transform():
    transform = A.Compose([
            ToTensorV2()
        ])
    
    return transform

