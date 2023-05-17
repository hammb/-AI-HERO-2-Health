from argparse import ArgumentParser
import torch
from torch import nn
#from torchmetrics.functional import dice_score
import numpy as np
import random
import pytorch_lightning as pl
from scipy import ndimage

from dataset import CellDataset, train_transform, val_transform
from unet2 import UNet
#from albumentations.pytorch.transforms import ToTensorV2


def seed_worker(worker_id):
    """
    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    to fix https://tanelp.github.io/posts/a-bug-that-plagues-thousands-of-open-source-ml-projects/
    ensures different random numbers each batch with each worker every epoch while keeping reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/s522r/Desktop/AIHERO/2.5/hackathon_health_data/train",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    #parser.add_argument("--augment", type=str, default="resize_rotate_crop")
    #parser.add_argument("--seed", type=int, default=42)
    '''parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )'''
    #parser.add_argument("--save_model", action="store_true", help="saves the trained model")
    #parser.add_argument("--model_name", type=str, help="model file name", default="unet_baseline")

    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir

    # Data
    train_data = CellDataset(root_dir, split="train", transform=train_transform())
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=16, shuffle=True, num_workers=12, worker_init_fn=seed_worker
    )
    val_data = CellDataset(root_dir, split="val", transform=val_transform())
    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=16, shuffle=False, num_workers=12, worker_init_fn=seed_worker
    )

    # Initialize the model and trainer
    model = UNet()
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.num_epochs, precision="16-mixed", benchmark=True)

    # Train the model   
    trainer.fit(model, trainloader, valloader)

