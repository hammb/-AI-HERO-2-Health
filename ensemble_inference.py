import torch
from dataset import CellDataset, val_transform
from deeplabv3_mobilenet_v3_large import DeepLab
from argparse import ArgumentParser
import os
import torch
from dataset import CellDataset, val_transform
from deeplabv3_mobilenet_v3_large import DeepLab
from argparse import ArgumentParser
from torch import nn
import cv2
import tifffile
import os
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg_mp
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train",
    )
    parser.add_argument("--from_checkpoints_dir", type=str, 
                        default='models')
    parser.add_argument("--pred_dir", default='./pred')
    parser.add_argument("--split", default="val", help="test")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split

    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Load the trained weights from all the checkpoints in the directory
    checkpoint_files = [file for file in os.listdir(args.from_checkpoints_dir) if file.endswith(".ckpt")]
    models = []

    for ckpt_file in checkpoint_files:
        model = DeepLab()
        checkpoint = torch.load(os.path.join(args.from_checkpoints_dir, ckpt_file))
        model.load_state_dict(checkpoint['state_dict'])
        models.append(model)

    # Ensemble the models and predict instances, then save them in the pred_dir
    with torch.no_grad():
        for batch, _, _, file_name in instance_seg_valloader:
            # Average the outputs of the models
            output_avg = sum([torch.softmax(model(batch), dim=1) for model in models]) / len(models)
            
            # Take the argmax to get the predicted class
            pred = torch.argmax(output_avg, 1)
            
            for i in range(pred.shape[0]):
                
                # convert to instance segmentation
                instance_segmentation = convert_semantic_to_instanceseg_mp(np.array(pred[i].unsqueeze(0).cpu()).astype(np.uint8), 
                                                                           spacing=(1, 1, 1), num_processes=12,
                                                                           isolated_border_as_separate_instance_threshold=15,
                                                                           small_center_threshold=30).squeeze()
                
                # resize to size 256x256
                resized_instance_segmentation = cv2.resize(instance_segmentation.astype(np.float32), (256,256), 
                           interpolation=cv2.INTER_NEAREST)                
                # save file 
                save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]
                os.makedirs(save_dir, exist_ok=True)
                tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_256.tif')), 
                                 resized_instance_segmentation.astype(np.uint16))
