# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Union

from mmengine import fileio
from mmengine.logging import MMLogger



from mmpretrain.registry import DATASETS

import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali import ops 

@DATASETS.register_module()
class DALIPipeline(Pipeline):
    def __init__(self, 
                 batch_size: int, 
                 num_threads: int, 
                 device_id, 
                 image_path, 
                 crop_size, 
                 shard_id, 
                 num_shards,
                 ann_file,
                 *kwargs):
        super(DALIPipeline, self).__init__(batch_size, num_threads, device_id, seed=12+device_id)
        # self.image_paths =self.read_image_paths_from_txt(ann_file)
        self.input = ops.readers.File(file_list=image_path, random_shuffle=True, shard_id=shard_id, num_shards=num_shards)
        
        self.decode = ops.decoders.Image(device='mixed', output_type=types.RGB)
        self.resized_crop = ops.RandomResizedCrop(
            device="gpu",
            size=(crop_size, crop_size),
            random_area=[0.5, 1.0])  # DALI uses LINEAR instead of BICUBIC
        self.flip = ops.Flip(horizontal=True,device='gpu')

    def define_graph(self):
        inputs, _ = self.input()
        images = self.decode(inputs)
        images = self.resized_crop(images)
        images = self.flip(images)
        return images
    
    def read_image_paths_from_txt(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        image_paths = [line.strip() for line in lines]
        return image_paths
