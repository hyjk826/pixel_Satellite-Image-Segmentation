import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from models.smp_models import UNet, DeepLabV3Plus
from loader.satellitedataset import SatelliteDataset
from torch.utils.data import DataLoader
import os
from loss.loss import DiceLoss, FocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scheduler.scheduler import CosineAnnealingWarmupRestarts

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def save_model(model, saved_dir, file_name='best_model.pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)
    
def dice_score(prediction: np.array, ground_truth: np.array, smooth=1e-7):
    intersection = np.sum(prediction * ground_truth)
    return (2.0 * intersection + smooth) / (np.sum(prediction) + np.sum(ground_truth) + smooth)

def calculate_dice(pred, gt):
    if np.sum(gt) > 0 or np.sum(pred) > 0:
        return dice_score(pred, gt)
    else:
        return None  # No valid masks found, return None
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    
def train():
    seed_everything(87)
    device = 'cuda:1'
    data_root = '/home/jovyan/work/datasets/satellite/'
    train_img_path = "images/train"
    train_anno_path = "annotations/train"
    valid_img_path = "images/validation"
    valid_anno_path = "annotations/validation"
    
    save_root = '/home/jovyan/work/work_space/uijin/baseline/work_dir/'
    test_name = "smpDeepLabV3Plus"
    
    model = DeepLabV3Plus()
    train_batch_size = 8
    valid_batch_size = 4
    epochs = 30
    save_interval = 5
    valid_interval = 1
    
    train_transform = A.Compose([  
                                A.RandomCrop(256, 256),
                                A.Resize(512, 512),
                                A.Normalize(),
                                ToTensorV2()
                                ])
    
    infer_transform = A.Compose([   
                                A.Resize(2048, 2048),
                                A.Normalize(),
                                ToTensorV2()
                                ])
    
    # loss function과 optimizer 정의
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = FocalLoss()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 1)
    
    # data setting
    train_dataset = SatelliteDataset(img_root=os.path.join(data_root, train_img_path),
                                    anno_root=os.path.join(data_root, train_anno_path),
                                    train_transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    
    valid_dataset = SatelliteDataset(img_root=os.path.join(data_root, valid_img_path),
                                    anno_root=os.path.join(data_root, valid_anno_path),
                                    train_transform=infer_transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=4)

    # training loop
    max_result = 0
    model = model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0
        print("train: [", epoch, "/", epochs, "] epoch", ", lr:", get_lr(optimizer))
        for images, masks in tqdm(train_dataloader):
            images = images.float().to(device)
            masks = masks.float().to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        print(f'Epoch {epoch}, train_Loss: {epoch_loss/len(train_dataloader)}')
        
        if epoch % save_interval == 0:
            save_model(model, os.path.join(save_root, test_name), 'epoch_' + epoch + '.pt')
            
        if epoch % valid_interval == 0:
            with torch.no_grad():
                model.eval()
                dice_scores = []
                for images, masks in tqdm(valid_dataloader):
                    images = images.float().to(device)
                    masks = masks.float().to(device)

                    outputs = model(images)
                    for output in outputs:
                        score = calcuate_dice(mask, output)
                        if score is not None:
                            dice_scores.append(score)
                result = np.mean(dice_scores)

                if result > max_result:
                    max_result = result
                    save_model(model, os.path.join(save_root, test_name))
                print("valid_result: ", result)

if __name__ == "__main__":
    train()