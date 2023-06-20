import torch
from dataset import CellDataset, val_transform
from unet import UNet
from argparse import ArgumentParser
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg_mp
import numpy as np
import cv2
import tifffile
import os

def predict_instance_segmentation_from_border_core(model1, model2, model3, dataloader, pred_dir='./preds'):
    # Set models to evaluation mode
    model1.eval()
    model2.eval()
    model3.eval()

    with torch.no_grad():  
        # Loop over all batches
        for batch, _, _, file_name in dataloader:
            # Pass the batch through the models and get the models' predictions
            pred1 = torch.argmax(model1(batch), 1)
            pred2 = torch.argmax(model2(batch), 1)
            pred3 = torch.argmax(model3(batch), 1)

            # Ensemble the predictions (here using average, adjust as needed)
            pred = (pred1 + pred2 + pred3) / 3.0
            pred = torch.argmax(pred, dim=1)

            # Loop over all predictions in the batch
            for i in range(pred.shape[0]):
                # Convert the predicted semantic segmentation to instance segmentation
                instance_segmentation = convert_semantic_to_instanceseg_mp(
                    np.array(pred[i].unsqueeze(0)).astype(np.uint8), 
                    spacing=(1, 1, 1), 
                    num_processes=12,
                    isolated_border_as_separate_instance_threshold=15,
                    small_center_threshold=30).squeeze()
                
                # Resize the instance segmentation to 256x256
                resized_instance_segmentation = cv2.resize(
                    instance_segmentation.astype(np.float32), 
                    (256,256), 
                    interpolation=cv2.INTER_NEAREST)                

                # Save the instance segmentation to disk
                save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]
                os.makedirs(save_dir, exist_ok=True)
                tifffile.imwrite(
                    os.path.join(save_dir, save_name.replace('.tif', '_256.tif')), 
                    resized_instance_segmentation.astype(np.uint64))


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

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split

    model = UNet()
    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Load the trained weights from the checkpoint
    checkpoint = torch.load(args.from_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    # predict instances and save them in the pred_dir
    # predict_instance_segmentation_from_border_core(model1, model2, model3, instance_seg_valloader, pred_dir=pred_dir)










