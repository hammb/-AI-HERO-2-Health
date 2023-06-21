import torch
from dataset import CellDataset, val_transform
from deeplabv3_mobilenet_v3_large import DeepLab
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/b556m/Downloads/pred_dir",
    )
    parser.add_argument("--from_checkpoint", type=str, 
                        default='./models/epoch=44-step=315.ckpt')
    parser.add_argument("--pred_dir", default='./pred')
    parser.add_argument("--split", default="test", help="test")
    
    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    pred_dir = args.pred_dir
    split = args.split

    model = DeepLab().half().to('cuda')
    instance_seg_val_data = CellDataset(root_dir, split=split, transform=val_transform(), border_core=False)
    instance_seg_valloader = torch.utils.data.DataLoader(
        instance_seg_val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Load the trained weights from the checkpoint
    checkpoint = torch.load(args.from_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    # predict instances and save them in the pred_dir
    model.predict_instance_segmentation_from_border_core(instance_seg_valloader, pred_dir=pred_dir)