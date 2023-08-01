import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm
from models.smp_models import UNet, DeepLabV3Plus
from loader.satellitedataset import SatelliteDataset, rle_encode
from torch.utils.data import DataLoader

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def inference():
    seed_everything(87)
    device = 'cuda'
    data_root = '/home/jovyan/work/prj_data/open'
    model = DeepLabV3Plus()
    model = model.to(device)
    # model = nn.DataParallel(model)
    ckpt_path = '/home/jovyan/work/work_space/uijin/baseline/work_dir/smpDeepLabV3Plus_res152/smpDeepLabV3Plus_res152_20ep.pt'
    save_path = '/home/jovyan/work/work_space/uijin/submit'
    file_name = 'baseline/smpDeepLabV3Plus_res152_20ep.csv'
    
    test_dataset = SatelliteDataset(data_root=data_root, infer=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    
    with torch.no_grad():
        model.eval()
        result = []
        for images in tqdm(test_dataloader):
            images = images.float().to(device)

            outputs = model(images)
            masks = torch.sigmoid(outputs).cpu().numpy()
            masks = np.squeeze(masks, axis=1)
            masks = (masks > 0.35).astype(np.uint8) # Threshold = 0.35

            for i in range(len(images)):
                mask_rle = rle_encode(masks[i])
                if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                    result.append(-1)
                else:
                    result.append(mask_rle)

    submit = pd.read_csv(os.path.join(save_path, 'sample_submission.csv'))
    submit['mask_rle'] = result
    submit.to_csv(os.path.join(save_path, file_name), index=False)

if __name__ == '__main__':
    inference()