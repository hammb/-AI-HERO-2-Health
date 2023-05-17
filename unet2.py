import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import cv2
from skimage.segmentation import watershed
from skimage import measure
import tifffile
import os
from torchmetrics.classification import JaccardIndex as IoU
from acvl_utils.instance_segmentation.instance_as_semantic_seg import convert_semantic_to_instanceseg_mp


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        return x


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConvBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    

class UNet(pl.LightningModule):
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        
        # Encoder
        self.encoder1 = DoubleConvBlock(in_channels, init_features)
        self.encoder2 = DoubleConvBlock(init_features, init_features*2)
        self.encoder3 = DoubleConvBlock(init_features*2, init_features*4)
        self.encoder4 = DoubleConvBlock(init_features*4, init_features*8)
        
        # Decoder
        self.decoder4 = UpConvBlock(init_features*8, init_features*4)
        self.decoder3 = UpConvBlock(init_features*4, init_features*2)
        self.decoder2 = UpConvBlock(init_features*2, init_features)
        self.decoder1 = nn.Conv2d(init_features, out_channels, kernel_size=1)

        # Metrics
        self.iou = IoU(task='multiclass', num_classes=3)

        # Loss
        #self.ce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool2d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(F.max_pool2d(enc2, kernel_size=2, stride=2))
        enc4 = self.encoder4(F.max_pool2d(enc3, kernel_size=2, stride=2))
        dec4 = self.decoder4(enc4, enc3)
        dec3 = self.decoder3(dec4, enc2)
        dec2 = self.decoder2(dec3, enc1)
        out = self.decoder1(dec2)
        return out

    def training_step(self, batch, batch_idx):
        inputs, labels, _ , _ = batch
        outputs = self(inputs)
        #loss = self.dice_loss(outputs, labels)
        loss = self.ce(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ , _ = batch
        outputs = self(inputs)
        #import IPython
        #IPython.embed()
        #loss = self.dice_loss(outputs, labels)
        loss = self.ce(outputs, labels)
        iou = self.iou(torch.argmax(outputs, 1), labels)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)

    '''def test_step(self, batch, batch_idx):
        inputs, labels, _ , _ = batch
        outputs = self(inputs)
        #loss = self.dice_loss(outputs, labels)
        loss = self.ce(outputs, labels)
        iou = self.iou((outputs > 0.5).int(), labels)
        self.log('test_loss', loss)
        self.log('test_iou', iou)'''

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def predict_instance_segmentation_from_border_core(self, dataloader, pred_dir='./preds'):
        self.eval()
        with torch.no_grad():
            
            for batch, _, orig_size, file_name in dataloader:
                # Pass the input tensor through the network to obtain the predicted output tensor
                pred = torch.argmax(self(batch), 1)

                for i in range(pred.shape[0]):
                    
                    '''vis = pred[i].numpy().astype(np.uint8)*125
                    desired_resolution = (1200, 1200)
                    vis = cv2.resize(vis, desired_resolution)
                    cv2.imshow('Sem Seg', vis)
                    cv2.waitKey(0)

                    continue'''
                    vis = pred[i].numpy().astype(np.uint8)
                    
                    instance_segmentation = convert_semantic_to_instanceseg_mp(np.array(pred[i].unsqueeze(0)).astype(np.uint8), 
                                                                               spacing=(1, 1, 1), num_processes=12,
                                                                               isolated_border_as_separate_instance_threshold=15,
                                                                               small_center_threshold=30).squeeze()
                    
                    # resize to original size
                    #print(instance_segmentation.max())
                    resized_instance_segmentation = cv2.resize(instance_segmentation.astype(np.float32), (256,256),#(orig_size[1][i].tolist(), orig_size[0][i].tolist()), 
                               interpolation=cv2.INTER_NEAREST)                
                    # save file 
                    save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]#.replace('.tif', '.nii.gz')
                    os.makedirs(save_dir, exist_ok=True)
                    #tifffile.imwrite(os.path.join(save_dir, save_name), resized_instance_segmentation.astype(np.uint64))
                    #tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_512.tif')), instance_segmentation.astype(np.uint64))
                    tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_256.tif')), resized_instance_segmentation.astype(np.uint64))

                    #tifffile.imwrite(os.path.join(save_dir, save_name.replace('.tif', '_bordercore.tif')), vis)
                #break
    
    def predict_instance_segmentation_watershed(self, dataloader, pred_dir='./preds'):

        self.eval()
        with torch.no_grad():
            #num = 0
            for batch, _, orig_size, file_name in dataloader:
                # Pass the input tensor through the network to obtain the predicted output tensor
                output_tensor = self(batch)

                # Convert the output tensor to a binary mask
                threshold = 0.5
                binary_mask = (output_tensor > threshold).squeeze().cpu().numpy().astype(np.uint8)

                #tifffile.imwrite('./dev/sem_seg_{}.tif'.format(num), binary_mask)
                #num += 1
                
                '''cv2.imshow('binary mask', binary_mask[-1]*255)
                cv2.waitKey(0)'''

                # Apply morphological operations to remove noise and smooth the edges of the mask
                kernel = np.ones((5, 5), np.uint8)#cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                eroded_binary_mask = cv2.erode(binary_mask, kernel, iterations=2)
                #eroded_binary_mask = cv2.dilate(eroded_binary_mask, kernel, iterations=2)

                '''cv2.imshow('eroded binary mask', eroded_binary_mask[-1]*255)
                cv2.waitKey(0)

                raise Exception'''

                '''import IPython
                IPython.embed()'''

                # Apply the watershed algorithm to obtain the instance segmentation
                for i in range(eroded_binary_mask.shape[0]):
                    distance_transform = cv2.distanceTransform(eroded_binary_mask[i], cv2.DIST_L2, 0)#5         
                    local_maxima = measure.label(distance_transform > 0.4 * distance_transform.max())#0.5
                    markers = cv2.connectedComponents(local_maxima.astype(np.uint8))[1]
                    
                    '''print(markers.max())
                    print(markers.shape)
                    cv2.imshow('markers', cv2.applyColorMap(cv2.convertScaleAbs(markers, alpha=(255.0/2147483647.0)), cv2.COLORMAP_HSV))
                    cv2.waitKey(0)'''

                    instance_segmentation = watershed(-distance_transform, markers, mask=binary_mask[i])

                    


                    '''label_vis = cv2.applyColorMap((cv2.convertScaleAbs(instance_segmentation, alpha=(255.0/2147483647.0)) * 10) % 255, cv2.COLORMAP_HSV)
                    cv2.imshow('Instance segmentation', label_vis)
                    cv2.waitKey(0)

                    raise Exception'''


                    # resize to original size
                    resized_instance_segmentation = cv2.resize(instance_segmentation, (orig_size[1][i].tolist(), orig_size[0][i].tolist()), 
                               interpolation=cv2.INTER_NEAREST)                
                    # save file 
                    save_dir, save_name = os.path.join(pred_dir, file_name[i].split('/')[0]), file_name[i].split('/')[1]#.replace('.tif', '.nii.gz')
                    os.makedirs(save_dir, exist_ok=True)
                    tifffile.imwrite(os.path.join(save_dir, save_name), resized_instance_segmentation.astype(np.uint64))
            
        





