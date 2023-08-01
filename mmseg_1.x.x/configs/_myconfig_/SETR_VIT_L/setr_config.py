_base_ = [
    'setr_naive.py',
    'schedule_80k.py', 
    'default_runtime_m.py',
    'cityscapes.py'
]
crop_size = (300, 300)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor, pretrained=None,
    # backbone=dict(
    #     drop_rate=0.,
    #     init_cfg=dict(
    #         type='Pretrained', checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_vit-large_8x1_768x768_80k_cityscapes/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth'),
    test_cfg=dict(mode='slide', crop_size=(300, 300), stride=(300, 300)))#크롭사이즈  768기본
    #test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(512, 512)))#크롭사이즈  768기본
optimizer = dict(weight_decay=0.0)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    paramwise_cfg=dict(custom_keys={'head': dict(lr_mult=10.)}))
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

#gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz
#checkpoint='https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_naive_vit-large_8x1_768x768_80k_cityscapes/setr_naive_vit-large_8x1_768x768_80k_cityscapes_20211123_000505-20728e80.pth')