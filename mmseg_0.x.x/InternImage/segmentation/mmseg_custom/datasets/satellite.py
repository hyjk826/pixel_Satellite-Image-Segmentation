# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module(force=True)
class satellite(CustomDataset):
    CLASSES = ('Background', 'building')
    PALETTE = [[0, 0, 0], [0, 255, 0]]
    label_map={0: 0, 1: 1}
    def __init__(self, **kwargs):
        super(satellite, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)