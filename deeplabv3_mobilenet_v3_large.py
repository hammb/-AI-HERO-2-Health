import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import cv2
import tifffile
import os
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg_mp
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.weight = weight if weight is not None else None
        self.device = device

    def forward(self, inputs, targets, smooth=1):
        # One-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float().to(self.device)

        targets_no_bg = targets_one_hot[:, 1:]

        # Apply sigmoid function to convert outputs into probabilities for soft Dice loss
        inputs_soft = torch.softmax(inputs, dim=1)[:, 1:]

        # Calculate Soft Dice Loss excluding background
        intersection = (inputs_soft * targets_no_bg).sum((0,1,2,3), keepdim=True)
        dice_score = (2. * intersection + smooth) / (inputs_soft.sum((0,1,2,3), keepdim=True) + targets_no_bg.sum((0,1,2,3), keepdim=True) + smooth)
        dice_loss = 1 - dice_score.mean()

        # Calculate Cross Entropy Loss
        CE_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)

        # Combine both losses
        combined_loss = 0.5 * CE_loss + 0.5 * dice_loss

        return combined_loss


class DeepLab(pl.LightningModule):
    def __init__(self, pretrained=True, num_classes=3, device='cuda'):
        super(DeepLab, self).__init__()
        if pretrained:
            self.model = deeplabv3_mobilenet_v3_large(pretrained=pretrained)
        else:
            self.model = deeplabv3_mobilenet_v3_large(weights=None)

        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

        # Metrics
        self.iou = IoU(task='multiclass', num_classes=num_classes, ignore_index=0)

        # Loss
        self.criterion = CombinedLoss(weight=torch.tensor([1.0, 1.0, 2.0]).to(device), device=device)

        # Adding a softmax layer
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model(x)['out']

    def training_step(self, batch, batch_idx):
        inputs, labels, _ , _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ , _ = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        iou = self.iou(torch.argmax(outputs, 1), labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def predict_instance_segmentation_from_border_core(self, dataloader, pred_dir='./preds'):
        self.eval()
        with torch.no_grad():
            
            for batch, _, _, file_name in dataloader:
                # Pass the input tensor through the network to obtain the predicted output tensor
                pred = torch.argmax(self(batch), 1)

                for i in range(pred.shape[0]):
                    
                    # convert to instance segmentation
                    instance_segmentation = convert_semantic_to_instanceseg_mp(np.array(pred[i].unsqueeze(0)).astype(np.uint8), 
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
                                     resized_instance_segmentation.astype(np.uint64))
