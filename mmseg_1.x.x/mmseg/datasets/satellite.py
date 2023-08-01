# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
from typing import Callable, Dict, List, Optional, Sequence, Union
import copy
import os.path as osp

import mmengine
import mmengine.fileio as fileio
import numpy as np
from mmengine.dataset import BaseDataset, Compose

@DATASETS.register_module()
class satellite(BaseSegDataset):
    METAINFO = dict(
                    classes=('Background', 'building'),
                    palette=[[0, 0, 0], [0, 255, 0]],
                    label_map={0: 0,
                              1: 1}
                    )
    # METAINFO = dict(
    #             classes=('building'),
    #             palette=[[0, 255, 0]]
    #             )
    
    
    def __init__(self,
                 ann_file: str = '',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 metainfo: Optional[dict] = None,
                 data_root: Optional[str] = None,
                 data_prefix: dict = dict(img_path='', seg_map_path=''),
                 filter_cfg: Optional[dict] = None,
                 indices: Optional[Union[int, Sequence[int]]] = None,
                 serialize_data: bool = True,
                 pipeline: List[Union[dict, Callable]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 ignore_index: int = 255,
                 reduce_zero_label: bool = None,
                 backend_args: Optional[dict] = None):
        super().__init__(ann_file,
                         img_suffix,
                         seg_map_suffix,
                         metainfo,
                         data_root,
                         data_prefix,
                         filter_cfg,
                         indices,
                         serialize_data,
                         pipeline,
                         test_mode,
                         lazy_init,
                         max_refetch,
                         ignore_index,
                         reduce_zero_label,
                         backend_args)
        
        
