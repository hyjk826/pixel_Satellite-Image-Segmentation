{\rtf1\ansi\ansicpg949\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import mmcv\
import torch\
from tqdm import tqdm\
# from mmcv.parallel import MMDataParallel\
from argparse import ArgumentParser\
from mmseg.apis import init_segmentor, inference_segmentor\
\
import pandas as pd\
import numpy as np\
import json\
import numpy as np\
\
# RLE \uc0\u51064 \u53076 \u46377  \u54632 \u49688 \
def rle_encode(mask):\
    pixels = mask.flatten()\
    pixels = np.concatenate([[0], pixels, [0]])\
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1\
    runs[1::2] -= runs[::2]\
    return ' '.join(str(x) for x in runs)\
\
def main(args):\
    save_dir = args.save_dir\
    file_name = args.file_name\
    config_file = args.config_file\
    checkpoint_file = args.checkpoint_file\
    sample_path = args.sample_path\
    test_image_path = args.test_image_path\
    device = args.device\
    print('save_dir : ', save_dir)\
    print('file_name : ', file_name)\
    print('config_file : ', config_file)\
    print('checkpoint_file : ', checkpoint_file)\
    print('sample_path : ', sample_path)\
    print('test_image_path : ', test_image_path)\
    # if args.config_file is None:\
    #     config_path = os.path.join('/home/jovyan/work/work_space/uijin/submit/mmseg/configs', file_name+".py")  \
    # if args.checkpoint_file is None:\
    #     ckpt_path = os.path.join('/home/jovyan/work/work_space/uijin/submit/mmseg/ckpts', file_name+".pth")\
        \
        \
    model = init_segmentor(config_file, checkpoint_file, device)\
    # model = MMDataParallel(model.cuda(), device_ids=[0])\
    \
    \
    data = pd.read_csv(sample_path)['img_id'].values.tolist()\
    \
    with torch.no_grad():\
        model.eval()\
        result = []\
        for img_id in tqdm(data):\
            img_path = os.path.join(test_image_path, img_id + ".png")\
            mask = inference_segmentor(model, img_path)\
            # mask = mask.pred_sem_seg.data\
            mask = np.array(mask)\
            mask_rle = rle_encode(mask)\
            if mask_rle == '': # \uc0\u50696 \u52769 \u46108  \u44148 \u47932  \u54589 \u49472 \u51060  \u50500 \u50696  \u50630 \u45716  \u44221 \u50864  -1\
                result.append(-1)\
            else:\
                result.append(mask_rle)\
\
    submit = pd.read_csv(sample_path)\
    submit['mask_rle'] = result\
    submit.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)\
\
def parse_args():\
    parser = ArgumentParser()\
    parser.add_argument('--save_dir', type=str, default = "/home/jovyan/work/work_space/JSJ/submit/mmseg")\
    parser.add_argument('--file_name', type=str, default = 'InternImage_Plus4_332000')\
    parser.add_argument('--config_file', type=str, default = '/home/jovyan/work/work_space/JSJ/InternImage/segmentation/work_dirs/InternImage/PLUS/Fourth/InternImage_plus.py')\
    parser.add_argument('--checkpoint_file', type=str, default = '/home/jovyan/work/work_space/JSJ/InternImage/segmentation/work_dirs/InternImage/best_mDice_iter_92000.pth')\
    parser.add_argument('--sample_path', type=str, default = "/home/jovyan/work/work_space/JSJ/submit/sample_submission.csv")\
    parser.add_argument('--test_image_path', type=str, default = "/home/jovyan/work/datasets/satellite/images/test")\
    parser.add_argument('--device', type=str, default = "cuda")\
    args = parser.parse_args()\
    return args\
\
if __name__ == '__main__':\
    args = parse_args()\
    main(args)}