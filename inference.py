import torch
from dataset import CellDataset, val_transform
from unet import UNet
from argparse import ArgumentParser


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
    model.predict_instance_segmentation_from_border_core(instance_seg_valloader, pred_dir=pred_dir)










