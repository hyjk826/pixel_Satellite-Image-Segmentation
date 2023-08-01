_base_ = [
    './fcn_hr_18.py', 'isaid.py',
    'default_runtime.py', 'schedule_80k-Copy1.py'
]

data_preprocessor = dict(size=(768, 768))
model = dict(
    data_preprocessor=data_preprocessor, decode_head=dict(num_classes=2))
model = dict(
    pretrained='open-mmlab://msra/hrnetv2_w48',
    backbone=dict(
        extra=dict(
            stage2=dict(num_channels=(48, 96)),
            stage3=dict(num_channels=(48, 96, 192)),
            stage4=dict(num_channels=(48, 96, 192, 384)))),
    decode_head=dict(
        in_channels=[48, 96, 192, 384], channels=sum([48, 96, 192, 384])))