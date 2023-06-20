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

def predict_instance_segmentation_from_border_core(model, dataloader, pred_dir='./preds'):
    model.eval()
    with torch.no_grad():
        
        for batch, _, _, file_name in dataloader:

            batch = batch.float()
            # Move the batch tensor to the same device as our model
            batch = batch.to(device)
            
            # Pass the input tensor through the network to obtain the predicted output tensor
            pred = torch.argmax(model(batch), 1)

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train",
    )
    parser.add_argument("--from_checkpoint", type=str, 
                        default='./lightning_logs/version_0/checkpoints/epoch=99-step=10000.ckpt')
    parser.add_argument("--pred_dir", default='./pred')
    parser.add_argument("--split", default="val", help="val=sequence c")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split

    model = DeepLab(pretrained=False).to(device)  # Added .to(device)
    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Load the trained weights from the checkpoint
    checkpoint = torch.load(args.from_checkpoint)

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 

    # 3. load the new state dict
    model.load_state_dict(model_dict)

    # predict instances and save them in the pred_dir
    predict_instance_segmentation_from_border_core(model, instance_seg_valloader, pred_dir=pred_dir)



