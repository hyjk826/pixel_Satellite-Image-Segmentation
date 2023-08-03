# InternImage

[Install](https://github.com/OpenGVLab/InternImage/blob/master/segmentation/README.md)

## Datasets
[Datasets](mmseg_0.x.x/segmentation/mmseg_custom/datasets/satellite.py)

## Pretrained Checkpoints

## How to start training

```
cd mmseg_0.x.x
python segmentation/train.py work_dirs/InternImage/m2f_config_k2.py
```

## How to start inference


# SWINv2

[Install](mmseg_1.x.x/docs/en/get_started.md)

## Configs

## pretrained path

Swin Pretrained pth  
[Download Link](https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth)


## Pretrained Checkpoints

swin90.19_second_Train_iter_320000.pth    
[Google Drive Link](https://drive.google.com/file/d/1fI4Zfn_rKznJbPGee37jaEuZb_bO8fDK/view?usp=share_link)


## How to start training

   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/swin/swin90.19_second_Train_config.py
   ```

## How to start inference

```
python work_dirs/swin/infer.py
```






# Mask2Former

[Install](mmseg_1.x.x/docs/en/get_started.md)

## Pretrained Checkpoints

1. m2f_K2_iter_90000.pth  
   [Google Drive Link](https://drive.google.com/file/d/1NHI02wH_hzVNtEsWXzRa23MlH903W7an/view?usp=share_link)

2. m2f_K3_iter_180000.pth    
   [Google Drive Link](https://drive.google.com/file/d/199AWAwd8n758zGUQIJlDVl_wd3dz1AwT/view?usp=share_link)

3. m2f_K4_iter_180000.pth  
   [Google Drive Link](https://drive.google.com/file/d/1C637rkVaIV14SmdTHMzFM4SYQp-ceY6e/view?usp=share_link)

## How to start training

1. K2 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k2.py
   ```
   
2. K3 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k3.py
   ```
   
3. K4 Training:  
   ```
   cd mmseg_1.x.x
   python tools/train.py work_dirs/mask2former/m2f_config_k4.py
   ```
## How to start inference

```
python work_dirs/mask2former/infer_m2f.py
```






- data_root
    - satellite3
        - kfold
            - fold_1
                - train
                    - 이미지파일1.jpg
                    - 이미지파일2.jpg
                    …
                - anno_train_img_gray
                    - 어노테이션1.jpg
                    - 어노테이션2.jpg
                    …
                - valid
                    - 이미지파일1.jpg
                    - 이미지파일2.jpg
                    …
                - anno_valid_img_gray
                    - 어노테이션1.jpg
                    - 어노테이션2.jpg
