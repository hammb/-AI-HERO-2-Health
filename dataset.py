from torch.utils.data import Dataset
import tifffile
from pathlib import Path
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2


class CellDataset(Dataset):
    def __init__(self, root_dir, border_core=True, split="train", transform=None):

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
          
        if self.border_core:
            if self.mask_files:
                # convert masks to border core representation where each instance gets label 1 for the core part of the 
                # instance and label 2 for the border part of the instance
                
                # Create an array to store eroded instances
                eroded_instances = np.zeros_like(mask)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # Define the erosion kernel
                for instance_label in np.unique(mask)[1:]:  # Exclude background label 0
                    instance_mask = (mask == instance_label).astype(np.uint8)
                    eroded_instance = cv2.erode(instance_mask, kernel, iterations=4)
                    eroded_instances += eroded_instance
                
                mask = np.where(eroded_instances==1, 1, 2*mask.clip(0, 1)) # 1 = core, 2 = border


        if self.transform is not None:
            if self.mask_files:
                transformed = self.transform(image=img, mask=mask)
            else:
                transformed = self.transform(image=img)
            img = transformed['image']
            if self.mask_files:
                mask = transformed['mask'].long()

        if self.mask_files and not self.border_core:
            mask = mask.unsqueeze(0)

        return img, mask, orig_size, file_name

    def __len__(self):
        return len(self.img_files)
    

def train_transform():
    transform = A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(0.5, 0.25),
            ToTensorV2()
        ])
    
    return transform

def val_transform():
    transform = A.Compose([
            A.Resize(512, 512, interpolation=cv2.INTER_NEAREST),
            A.Normalize(0.5, 0.25),
            ToTensorV2()
        ])
    
    return transform

