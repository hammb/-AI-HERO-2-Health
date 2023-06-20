import os
import cv2
import tifffile
import numpy as np
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def process_files(img_file, mask_file, save_img_dir, save_mask_dir):
    # Load mask and convert it to float32
    mask = tifffile.imread(mask_file).astype(np.float32)

    # Create an array to store eroded instances
    eroded_instances = np.zeros_like(mask)

    # Define the erosion kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Iterate over each unique instance (excluding background label 0)
    for instance_label in np.unique(mask)[1:]:
        instance_mask = (mask == instance_label).astype(np.uint8)
        eroded_instance = cv2.erode(instance_mask, kernel, iterations=4)
        eroded_instances += eroded_instance

    # Get the new mask (1 = core, 2 = border)
    new_mask = np.where(eroded_instances==1, 1, 2*mask.clip(0, 1))

    # Save the new mask to disk (set appropriate filename)
    save_mask_file = save_mask_dir / mask_file.name
    save_mask_file.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_mask_file, new_mask)

    # Copy the image file to the new directory
    save_img_file = save_img_dir / img_file.name
    save_img_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(img_file, save_img_file)

def convert_to_border_core_representation(root_dir, save_dir):
    root_dir = Path(root_dir)
    save_dir = Path(save_dir)

    # Check if save_dir exists, if not, create it
    if not save_dir.exists():
        os.makedirs(save_dir)

    splits = ['a', 'b', 'c']
    for split in splits:
        img_dir = root_dir / split
        mask_dir = root_dir / f"{split}_GT"
        save_mask_dir = save_dir / f"{split}_GT"
        save_img_dir = save_dir / split

        img_files = sorted(list(img_dir.glob("*.tif")))
        mask_files = sorted(list(mask_dir.glob("*.tif")))

        with ThreadPoolExecutor() as executor:
            executor.map(process_files, img_files, mask_files, [save_img_dir]*len(img_files), [save_mask_dir]*len(mask_files))

# call the function
convert_to_border_core_representation('/path/to/your/original/data', '/path/to/save/new/masks')
