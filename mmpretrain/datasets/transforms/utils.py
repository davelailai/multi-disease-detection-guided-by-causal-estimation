# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Union

from mmcv.transforms import BaseTransform

PIPELINE_TYPE = List[Union[dict, BaseTransform]]


def get_transform_idx(pipeline: PIPELINE_TYPE, target: str) -> int:
    """Returns the index of the transform in a pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.

    Returns:
        int: The transform index. Returns -1 if not found.
    """
    for i, transform in enumerate(pipeline):
        if isinstance(transform, dict):
            if isinstance(transform['type'], type):
                if transform['type'].__name__ == target:
                    return i
            else:
                if transform['type'] == target:
                    return i
        else:
            if transform.__class__.__name__ == target:
                return i

    return -1


def remove_transform(pipeline: PIPELINE_TYPE, target: str, inplace=False):
    """Remove the target transform type from the pipeline.

    Args:
        pipeline (List[dict] | List[BaseTransform]): The transforms list.
        target (str): The target transform class name.
        inplace (bool): Whether to modify the pipeline inplace.

    Returns:
        The modified transform.
    """
    idx = get_transform_idx(pipeline, target)
    if not inplace:
        pipeline = copy.deepcopy(pipeline)
    while idx >= 0:
        pipeline.pop(idx)
        idx = get_transform_idx(pipeline, target)

    return pipeline


import random
import warnings
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data._utils.collate import \
    default_collate as torch_default_collate

from mmengine.registry import FUNCTIONS
from mmengine.structures import BaseDataElement
from mmengine.dataset import default_collate

# FUNCTIONS is new in MMEngine v0.7.0. Reserve the `COLLATE_FUNCTIONS` to keep
# the compatibility.
def torch_custom_collate(batch):
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        # Concatenate tensors along the batch dimension
        return torch.cat(batch, dim=0)
    elif isinstance(batch[0], list):
        # If elements are lists, concatenate them element-wise
        return [torch_custom_collate(samples) for samples in zip(*batch)]
    elif isinstance(batch[0], dict):
        # If elements are dictionaries, collate each key separately
        return {key: torch_custom_collate([d[key] for d in batch]) for key in batch[0]}
    else:
        raise TypeError(f"Unsupported type for collation: {elem_type}")
    
COLLATE_FUNCTIONS = FUNCTIONS
@FUNCTIONS.register_module()
def custom_collate(data_batch: Sequence) -> Any:
    """Convert list of data sampled from dataset into a batch of data, of which
    type consistent with the type of each data_itement in ``data_batch``.

    Different from :func:`pseudo_collate`, ``default_collate`` will stack
    tensor contained in ``data_batch`` into a batched tensor with the
    first dimension batch size, and then move input tensor to the target
    device.

    Different from ``default_collate`` in pytorch, ``default_collate`` will
    not process ``BaseDataElement``.

    This code is referenced from:
    `Pytorch default_collate <https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py>`_.

    Note:
        ``default_collate`` only accept input tensor with the same shape.

    Args:
        data_batch (Sequence): Data sampled from dataset.

    Returns:
        Any: Data in the same format as the data_itement of ``data_batch``, of which
        tensors have been stacked, and ndarray, int, float have been
        converted to tensors.
    """  # noqa: E501
    data_item = data_batch[0]
    data_item_type = type(data_item)

    if isinstance(data_item, (BaseDataElement, str, bytes)):
        return data_batch
    elif isinstance(data_item, tuple) and hasattr(data_item, '_fields'):
        # named_tuple
        return data_item_type(*(custom_collate(samples)
                                for samples in zip(*data_batch)))
    elif isinstance(data_item, Sequence):
        # check to make sure that the data_itements in batch have
        # consistent size
        it = iter(data_batch)
        data_item_size = len(next(it))
        if not all(len(data_item) == data_item_size for data_item in it):
            raise RuntimeError(
                'each data_itement in list of batch should be of equal size')
        transposed = list(zip(*data_batch))

        if isinstance(data_item, tuple):
            return [custom_collate(samples)
                    for samples in transposed]  # Compat with Pytorch.
        else:
            try:
                return data_item_type(
                    [custom_collate(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [custom_collate(samples) for samples in transposed]
    elif isinstance(data_item, Mapping):
        index = data_item_type({
            key: custom_collate([d[key] for d in data_batch])
            for key in data_item
        })
        return data_item_type({
            key: custom_collate([d[key] for d in data_batch])
            for key in data_item
        })
    else:
        return torch_custom_collate(data_batch)