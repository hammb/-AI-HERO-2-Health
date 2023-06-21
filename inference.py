import torch
from dataset import CellDataset, test_transform
from deeplabv3_mobilenet_v3_large import ResDeepLab
from argparse import ArgumentParser
import glob
import torch
import numpy as np
import cv2
import tifffile
import os
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg_mp
from postprocessing import post_process

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-health_train_data/train",
    )
    parser.add_argument("--from_checkpoint", type=str, 
                        default='/hkfs/work/workspace/scratch/hgf_pdv3669-H2/-AI-HERO-2-Health/models/submission_res') # must be directory
    parser.add_argument("--pred_dir", default='./pred_res')
    parser.add_argument("--split", default="val", help="choose the split (train, val, test)")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split
    checkpoint_dir = args.from_checkpoint
    assert os.path.isdir(checkpoint_dir), "checkpoint_dir must be a directory"

    batch_size = 128

    model = ResDeepLab().half().to('cuda')
    instance_seg_test_data = CellDataset(root_dir, split=split, transform=test_transform(), border_core=False)
    instance_seg_testloader = torch.utils.data.DataLoader(
        instance_seg_test_data, batch_size=batch_size, shuffle=False, num_workers=32
    )

    # find all checkpoints in the checkpoint_dir
    checkpoints = glob.glob(checkpoint_dir + "/*.ckpt")

    for batch, _, _, file_name in instance_seg_testloader:
        logits = []

        for idx, checkpoint in enumerate(checkpoints):
            model.load_state_dict(torch.load(checkpoint)["state_dict"])
            logits.append(model.predict_border_core_logits(batch))

        # Average the logits 
        pred = torch.argmax(torch.mean(torch.stack(logits), dim=0), 1)

        # convert to instance segmentation
        for i in range(pred.shape[0]):      
            # convert to instance segmentation
            instance_segmentation = convert_semantic_to_instanceseg_mp(np.array(pred[i].cpu().unsqueeze(0)).astype(np.uint8), 
                                                                        spacing=(1, 1, 1), num_processes=32,
                                                                        isolated_border_as_separate_instance_threshold=15,
                                                                        small_center_threshold=30).squeeze()
            
            # resize to size 256x256
            resized_instance_segmentation = cv2.resize(instance_segmentation.astype(np.float32), (256,256), 
                        interpolation=cv2.INTER_NEAREST)

            # Post processing
            postproc_seg = post_process(resized_instance_segmentation, threshold=50)
                          
            # save file 
            save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]
            os.makedirs(save_dir, exist_ok=True)
            tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_256.tif')), 
                                postproc_seg.astype(np.uint16))