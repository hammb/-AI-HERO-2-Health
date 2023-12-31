from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import CellDataset, train_transform, val_transform
from unet import UNet
from deeplabv3_mobilenet_v3_large import DeepLab, ResDeepLab
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/hkfs/work/workspace/scratch/hgf_pdv3669-H2/data/preprocessed_baseline_rn",
    )
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--fold", type=str, default="c")
    parser.add_argument("--model", type=str, default="deeplab", choices=["deeplab", "resdeeplab"])

    args = parser.parse_args()

    device = torch.device("cuda")
    root_dir = args.root_dir
    fold = args.fold

    # Data
    train_data = CellDataset(root_dir, split="train", fold=fold, transform=train_transform())
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True, num_workers=64
    )
    val_data = CellDataset(root_dir, split="val", fold=fold, transform=val_transform())
    valloader = torch.utils.data.DataLoader(
        val_data, batch_size=32, shuffle=False, num_workers=64
    )

    # Initialize the model and trainer
    if args.model == "deeplab":
        model = DeepLab()
    elif args.model == "resdeeplab":
        model = ResDeepLab()
    else:    
        raise ValueError("Model not supported")
    
    # Logging, write to disk after 10 logging events or every two minutes
    logger = TensorBoardLogger("logs/", name="ai_hero", max_queue=10, flush_secs=120)

    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.0, patience=10, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(dirpath='logs/ai_hero_res/', filename='ai_hero-{epoch:02d}-{val_loss:.2f}-'+f'fold-{fold}')

    trainer = pl.Trainer(accelerator='gpu',
                         strategy="ddp",
                         devices=4, 
                         max_epochs=args.num_epochs,
                         precision="16-mixed",
                         benchmark=True,
                         logger=logger,
                         callbacks=[early_stop_callback, checkpoint_callback])

    # Train the model   
    trainer.fit(model, trainloader, valloader)

