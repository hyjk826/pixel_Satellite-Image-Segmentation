import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from tqdm import tqdm
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral
from skimage.color import gray2rgb
from skimage.color import rgb2gray
from ipywidgets import widgets, interact

# RLE 디코딩 함수
def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# RLE 인코딩 함수
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def use_crf(image, mask, crf_steps=10, gt_prob=0.76):
    # Converting annotated image to RGB if it is Gray scale
    annotated_label = mask.flatten()
    if(len(mask.shape)<3):
        mask = gray2rgb(mask)

    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = mask[:,:,0] + (mask[:,:,1]<<8) + (mask[:,:,2]<<16)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)

    n_labels = 2
    
    #Setting up the CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    U = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                      normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for n steps 
    Q = d.inference(crf_steps)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    return MAP.reshape((image.shape[0],image.shape[1]))

def main():
    save_dir = "/home/jovyan/work/work_space/uijin/submit/"
    file_name = "dense_crf"
    target_csv = "/home/jovyan/work/work_space/uijin/util/target_82163.csv"
    CRF_STEPS = 10
    GT_PROB = 0.78
            
    test_img_dir = "/home/jovyan/work/prj_data/open/test_img"
    sample_path = "/home/jovyan/work/work_space/uijin/submit/sample_submission.csv"
    table = pd.read_csv(target_csv)
    result = []
    for img_id in tqdm(table["img_id"]):
        mask_rle = table[table["img_id"]==img_id]["mask_rle"].values[0]
        img_path = os.path.join(test_img_dir, img_id + ".png")
        img = cv2.imread(img_path)
        mask = rle_decode(mask_rle, (img.shape[0], img.shape[1]))
        mask_afterCRF = use_crf(img, mask, crf_steps=CRF_STEPS, gt_prob=GT_PROB)
        mask_rle_afterCRF = rle_encode(mask_afterCRF)
        
        if mask_rle_afterCRF == '': # 예측된 건물 픽셀이 아예 없는 경우 -1
            result.append(-1)
        else:
            result.append(mask_rle_afterCRF)
            
    submit = pd.read_csv(sample_path)
    submit['mask_rle'] = result
    submit.to_csv(os.path.join(save_dir, file_name + '.csv'), index=False)
        
if __name__ == '__main__':
    main()
    