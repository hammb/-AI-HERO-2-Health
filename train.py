from argparse import ArgumentParser
import torch
import pytorch_lightning as pl

from dataset import CellDataset, train_transform, val_transform
from unet import UNet


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/s522r/Desktop/AIHERO/2.5/hackathon_health_data/train",
    )
    parser.add_argument("--num_epochs", type=int, default=100)

    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir

    # Data
    train_data = CellDataset(root_dir, split="train", transform=train_transform())
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=16, shuffle=True, num_workers=12
    )
    val_data = CellDataset(root_dir, split="val", transform=val_transform())
    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=16, shuffle=False, num_workers=12
    )

    # Initialize the model and trainer
    model = UNet()
    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=args.num_epochs, precision="16-mixed", benchmark=True)

    # Train the model   
    trainer.fit(model, trainloader, valloader)

