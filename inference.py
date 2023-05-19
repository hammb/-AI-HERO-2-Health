import torch
from dataset import CellDataset, val_transform
from unet import UNet
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/s522r/Desktop/AIHERO/2.5/hackathon_health_data/train",
    )
    parser.add_argument("--from_checkpoint", type=str, 
                        default='./lightning_logs/version_80/checkpoints/epoch=99-step=10000.ckpt')
    parser.add_argument("--pred_dir", default='./pred')
    
    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir

    model = UNet()
    instance_seg_val_data = CellDataset(root_dir, split="val", transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Load the trained weights from the checkpoint
    checkpoint = torch.load(args.from_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    # predict instances and save them in the pred_dir
    model.predict_instance_segmentation_from_border_core(instance_seg_valloader, pred_dir=pred_dir)










