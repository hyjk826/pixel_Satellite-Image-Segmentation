import os
import mmcv
import torch
from tqdm import tqdm

from argparse import ArgumentParser
from mmseg.apis import init_model, inference_model

import pandas as pd
import numpy as np
import json
import numpy as np

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def main(args):
    save_dir = args.save_dir
    file_name = args.file_name
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    sample_path = args.sample_path
    test_image_path = args.test_image_path
    device = args.device
    

    model = init_model(config_path, ckpt_path, device)

    
    data = pd.read_csv(sample_path)['img_id'].values.tolist()
    
    with torch.no_grad():
        model.eval()
        result = []
        for img_id in tqdm(data):
            img_path = os.path.join(test_image_path, img_id + ".png")
            mask = inference_model(model, img_path)
            mask = mask.pred_sem_seg.data
            mask =torch.squeeze(mask).cpu().numpy()
            
            mask_rle = rle_encode(mask)
            if mask_rle == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
                result.append(-1)
            else:
                result.append(mask_rle)

    submit = pd.read_csv(sample_path)
    submit['mask_rle'] = result
    submit.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--save_dir', type=str, default = ".") #save dir
    parser.add_argument('--file_name', type=str, default = 'submit_m2f.csv')
    parser.add_argument('--config_path', type=str, default = 'work_dirs/mask2former/mask2former_config_k3.py')
    #k2 | k3 | k4
    parser.add_argument('--ckpt_path', type=str, default = 'work_dirs/mask2former/m2f_K3_iter_90000.pth')
    #드라이브에서 다운 후 지정
    parser.add_argument('--sample_path', type=str, default = "work_dirs/sample_submission.csv")
    parser.add_argument('--test_image_path', type=str, default = "/home/jovyan/work/datasets/satellite2/test") 
    #test image dir
    parser.add_argument('--device', type=str, default = "cuda")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)