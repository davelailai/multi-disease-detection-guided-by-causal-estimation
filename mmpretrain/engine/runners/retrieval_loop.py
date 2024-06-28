# Copyright (c) OpenMMLab. All rights reserved.
import copy
from mmengine.model import is_model_wrapper
from mmengine.runner import TestLoop, ValLoop, autocast,Runner, _get_batch_size, set_random_seed
import logging
# from mmengine.runner import *

import torch.nn as nn
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval

from mmengine.runner.activation_checkpointing import turn_on_activation_checkpointing

from mmpretrain.registry import LOOPS, RUNNERS,PARAM_SCHEDULERS
from mmengine.runner.base_loop import BaseLoop
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)

from mmengine.runner.loops import EpochBasedTrainLoop, IterBasedTrainLoop, TestLoop, ValLoop
from torch.utils.data import DataLoader
import torch
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union
from mmengine.registry import DATASETS,FUNCTIONS
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator,DALIClassificationIterator
from nvidia.dali import ops

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, get_world_size,
                           init_dist, is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)

@pipeline_def
def file_pipeline(files, crop_size, shard_id, num_shards):

    # device_id = Pipeline.current().device_id
    crop=ops.RandomResizedCrop(
            device="gpu",
            size=(crop_size, crop_size),
            random_area=[0.5, 1.0]) 
    flip = ops.Flip(horizontal=True,device='gpu')

    # device_id = Pipeline.current().device_id
    transpose = ops.Transpose(device="gpu", perm=[2, 0, 1])

    jpegs, _ = fn.readers.file(
        files=files,
        random_shuffle=True,
        shard_id=shard_id,
        num_shards=num_shards,
        pad_last_batch=True,
        name="Reader")
    images = fn.decoders.image(jpegs, device="mixed")
    images=crop(images)
    images=flip(images)
    images=transpose(images)

    return images, images


@LOOPS.register_module()
class RetrievalValLoop(ValLoop):
    """Loop for multimodal retrieval val.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 valing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch val."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_val_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor

                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_val_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = {
            k: torch.cat([dic[k] for dic in feats_local])
            for k in feats_local[0]
        }

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
            )

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        i2t_metrics = self.evaluator.evaluate(img_size)
        i2t_metrics = {f'i2t/{k}': v for k, v in i2t_metrics.items()}
        self.evaluator.process(t2i_data_samples, None)
        t2i_metrics = self.evaluator.evaluate(text_size)
        t2i_metrics = {f't2i/{k}': v for k, v in t2i_metrics.items()}
        metrics = {**i2t_metrics, **t2i_metrics}

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics


@LOOPS.register_module()
class RetrievalTestLoop(TestLoop):
    """Loop for multimodal retrieval test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        feats_local = []
        data_samples_local = []

        for idx, data_batch in enumerate(self.dataloader):
            with torch.no_grad():
                self.runner.call_hook(
                    'before_test_iter', batch_idx=idx, data_batch=data_batch)
                # predictions should be sequence of BaseDataElement
                with autocast(enabled=self.fp16):
                    if is_model_wrapper(self.runner.model):
                        data_preprocessor = self.runner.model.module.data_preprocessor  # noqa: E501
                    else:
                        data_preprocessor = self.runner.model.data_preprocessor
                    # get features for retrieval instead of data samples
                    data_batch = data_preprocessor(data_batch, False)
                    feats = self.runner.model._run_forward(
                        data_batch, mode='tensor')
                    feats_local.append(feats)
                    data_samples_local.extend(data_batch['data_samples'])
                self.runner.call_hook(
                    'after_test_iter',
                    batch_idx=idx,
                    data_batch=data_batch,
                    outputs=feats)

        # concatenate different features
        feats_local = {
            k: torch.cat([dic[k] for dic in feats_local])
            for k in feats_local[0]
        }

        # get predictions
        if is_model_wrapper(self.runner.model):
            predict_all_fn = self.runner.model.module.predict_all
        else:
            predict_all_fn = self.runner.model.predict_all

        img_size = self.dataloader.dataset.img_size
        text_size = self.dataloader.dataset.text_size
        with torch.no_grad():
            i2t_data_samples, t2i_data_samples = predict_all_fn(
                feats_local,
                data_samples_local,
                num_images=img_size,
                num_texts=text_size,
            )

        # process in evaluator and compute metrics
        self.evaluator.process(i2t_data_samples, None)
        i2t_metrics = self.evaluator.evaluate(img_size)
        i2t_metrics = {f'i2t/{k}': v for k, v in i2t_metrics.items()}
        self.evaluator.process(t2i_data_samples, None)
        t2i_metrics = self.evaluator.evaluate(text_size)
        t2i_metrics = {f't2i/{k}': v for k, v in t2i_metrics.items()}
        metrics = {**i2t_metrics, **t2i_metrics}

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

@LOOPS.register_module()

@RUNNERS.register_module()
class DALI_runner(Runner):

    
    @staticmethod
    def build_dataloader(dataloader,
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False):
        dataloader_cfg = copy.deepcopy(dataloader)
        world_size = get_world_size()
        batch_size_per_gpu = _get_batch_size(dataloader_cfg)
        local_rank= get_rank()
        # num_gpus = torch.cuda.device_count()
        print(local_rank)
        # print(world_size)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        # dataset_cfg['batch_size'] = batch_size_per_gpu * num_gpus
        dataset_cfg['device_id'] = local_rank
        dataset_cfg['shard_id'] =local_rank
        dataset_cfg['num_shards'] =world_size
        if 'num_threads' in dataloader:
            dataset_cfg['num_threads'] = dataloader['num_threads']
        
        with open(dataset_cfg['ann_file'], 'r') as file:
            lines = file.readlines()
        image_paths = [line.strip() for line in lines]
        # pipes = []
        # if isinstance(dataset_cfg, dict):
        #     for device_id in range(world_size):
        #         dataset_cfg['device_id'] = device_id
        #         dataset_cfg['shard_id'] = device_id
        #         dataset_cfg['shard_id'] = device_id
        #         dataset_cfg['image_path'] = image_paths
                # pipes.append(DATASETS.build(dataset_cfg))
        # pipes=[]
        # for device_id in range(world_size):
        #     pipe = file_pipeline(
        #                         batch_size=batch_size_per_gpu, 
        #                         num_threads=dataloader['num_threads'], 
        #                         py_num_workers=dataloader['num_workers'],
        #                         seed=12 + local_rank,
        #                         files=image_paths,
        #                         crop_size = dataset_cfg['crop_size'], 
        #                         num_shards=world_size,
        #                         device_id=device_id)

        # for pipe in pipes:
        #     pipe.build()

        pipes = file_pipeline(
                            batch_size=batch_size_per_gpu, 
                            num_threads=dataloader['num_threads'], 
                            py_num_workers=dataloader['num_workers'],
                            seed=12 + local_rank,
                            device_id=local_rank,
                            files=image_paths,
                            crop_size = dataset_cfg['crop_size'], 
                            num_shards=world_size,
                            shard_id=local_rank)

        pipes.build()

        # pipe.build()
        # pipe.set_outputs

        # dali_iter = DALIClassificationIterator(pipe, reader_name="Reader", output_map=['input','Datasample'])
        dali_iter = DALIGenericIterator(pipes, output_map=['input','Datasample'],reader_name="Reader")
        dali_iter.dataset= dataset_cfg

        # for idx, data_batch in enumerate(dali_iter):
        #     data_batch

        return dali_iter
    
    def len_dataset(self):
        import math
        image_file=self._train_dataloader['dataset']['ann_file']
        with open(image_file, 'r') as file:
            lines = file.readlines()
        total_num=len(lines)

        batch_size=self._train_dataloader['batch_size']
        num_gpu=self._world_size

        return math.ceil(total_num / (batch_size * num_gpu))

    def build_train_loop(self, loop: Union[BaseLoop, Dict]) -> BaseLoop:
        """Build training loop.

        Examples of ``loop``::

            # `EpochBasedTrainLoop` will be used
            loop = dict(by_epoch=True, max_epochs=3)

            # `IterBasedTrainLoop` will be used
            loop = dict(by_epoch=False, max_epochs=3)

            # custom training loop
            loop = dict(type='CustomTrainLoop', max_epochs=3)

        Args:
            loop (BaseLoop or dict): A training loop or a dict to build
                training loop. If ``loop`` is a training loop object, just
                returns itself.

        Returns:
            :obj:`BaseLoop`: Training loop object build from ``loop``.
        """
        if isinstance(loop, BaseLoop):
            return loop
        elif not isinstance(loop, dict):
            raise TypeError(
                f'train_loop should be a Loop object or dict, but got {loop}')

        loop_cfg = copy.deepcopy(loop)

        if 'type' in loop_cfg and 'by_epoch' in loop_cfg:
            raise RuntimeError(
                'Only one of `type` or `by_epoch` can exist in `loop_cfg`.')

        if 'type' in loop_cfg:
            loop = LOOPS.build(
                loop_cfg,
                default_args=dict(
                    runner=self, dataloader=self._train_dataloader))
        else:
            by_epoch = loop_cfg.pop('by_epoch')
            if by_epoch:
                loop = EpochBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
            else:
                loop = IterBasedTrainLoop(
                    **loop_cfg, runner=self, dataloader=self._train_dataloader)
        return loop  # type: ignore
    
    def _build_param_scheduler(
            self, scheduler: Union[_ParamScheduler, Dict, List],
            optim_wrapper: OptimWrapper) -> List[_ParamScheduler]:
        """Build parameter schedulers for a single optimizer.

        Args:
            scheduler (_ParamScheduler or dict or list): A Param Scheduler
                object or a dict or list of dict to build parameter schedulers.
            optim_wrapper (OptimWrapper): An optimizer wrapper object is
                passed to construct ParamScheduler object.

        Returns:
            list[_ParamScheduler]: List of parameter schedulers build from
            ``scheduler``.

        Note:
            If the train loop is built, when building parameter schedulers,
            it supports setting the max epochs/iters as the default ``end``
            of schedulers, and supports converting epoch-based schedulers
            to iter-based according to the ``convert_to_iter_based`` key.
        """
        if not isinstance(scheduler, Sequence):
            schedulers = [scheduler]
        else:
            schedulers = scheduler

        param_schedulers = []
        for scheduler in schedulers:
            if isinstance(scheduler, _ParamScheduler):
                param_schedulers.append(scheduler)
            elif isinstance(scheduler, dict):
                _scheduler = copy.deepcopy(scheduler)

                # Set default end
                if isinstance(self._train_loop, BaseLoop):
                    default_end = self.max_epochs if _scheduler.get(
                        'by_epoch', True) else self.max_iters
                    _scheduler.setdefault('end', default_end)
                    self.logger.debug(
                        f'The `end` of {_scheduler["type"]} is not set. '
                        'Use the max epochs/iters of train loop as default.')

                param_schedulers.append(
                    PARAM_SCHEDULERS.build(
                        _scheduler,
                        default_args=dict(
                            optimizer=optim_wrapper,
                            epoch_length=self.len_dataset())))
            else:
                raise TypeError(
                    'scheduler should be a _ParamScheduler object or dict, '
                    f'but got {scheduler}')
        return param_schedulers

    def train(self) -> nn.Module:
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        # `build_optimizer` should be called before `build_param_scheduler`
        #  because the latter depends on the former
        self.optim_wrapper = self.build_optim_wrapper(self.optim_wrapper)
        # Automatically scaling lr by linear scaling rule
        self.scale_lr(self.optim_wrapper, self.auto_scale_lr)

        if self.param_schedulers is not None:
            self.param_schedulers = self.build_param_scheduler(  # type: ignore
                self.param_schedulers)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()

        # try to enable activation_checkpointing feature
        modules = self.cfg.get('activation_checkpointing', None)
        if modules is not None:
            self.logger.info(f'Enabling the "activation_checkpointing" feature'
                             f' for sub-modules: {modules}')
            turn_on_activation_checkpointing(ori_model, modules)

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Initiate inner count of `optim_wrapper`.
        self.optim_wrapper.initialize_count_status(
            self.model,
            self._train_loop.iter,  # type: ignore
            self._train_loop.max_iters)  # type: ignore

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model

class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.file_path = file_path
        self.transform = transform
        self.image_paths = self.load_image_paths()

    def load_image_paths(self):
        with open(self.file_path, 'r') as file:
            image_paths = [line.strip() for line in file.readlines()]
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def collate_fn(batch):
    # Assume each item in batch is a tuple (input_tensor, target_tensor)
    # for item in batch:
    #     inputs=item['inputs']
    inputs = [item['inputs'] for item in batch]

    # N,B,C,H,W = data_batch['inputs'].shape
    # data_batch['inputs']=data_batch['inputs'].reshape(-1,C,H,W)
    # targets = [item[1] for item in batch]

    # Stack inputs and targets along the batch dimension
    collated_inputs=[]
    collated_inputs['inputs'] = torch.cat(inputs,dim=0)
    collated_inputs['data_samples'] = inputs[0]['data_samples']



    return collated_inputs

class _SlicedDataset:

    def __init__(self, dataset, length) -> None:
        self._dataset = dataset
        self._length = length

    def __getattr__(self, name):
        return getattr(self._dataset, name)

    def __getitem__(self, idx):
        return self._dataset[idx]

    def __len__(self):
        return self._length

@RUNNERS.register_module()
class torchvision_runner(Runner):

    @staticmethod
    def build_dataloader(dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        """Build dataloader.

        The method builds three components:

        - Dataset
        - Sampler
        - Dataloader

        An example of ``dataloader``::

            dataloader = dict(
                dataset=dict(type='ToyDataset'),
                sampler=dict(type='DefaultSampler', shuffle=True),
                batch_size=1,
                num_workers=9
            )

        Args:
            dataloader (DataLoader or dict): A Dataloader object or a dict to
                build Dataloader object. If ``dataloader`` is a Dataloader
                object, just returns itself.
            seed (int, optional): Random seed. Defaults to None.
            diff_rank_seed (bool): Whether or not set different seeds to
                different ranks. If True, the seed passed to sampler is set
                to None, in order to synchronize the seeds used in samplers
                across different ranks.


        Returns:
            Dataloader: DataLoader build from ``dataloader_cfg``.
        """
        if isinstance(dataloader, DataLoader):
            return dataloader

        dataloader_cfg = copy.deepcopy(dataloader)

        # build dataset
        dataset_cfg = dataloader_cfg.pop('dataset')
        if isinstance(dataset_cfg, dict):
            dataset = DATASETS.build(dataset_cfg)
            if hasattr(dataset, 'full_init'):
                dataset.full_init()
        else:
            # fallback to raise error in dataloader
            # if `dataset_cfg` is not a valid type
            dataset = dataset_cfg

        num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
        if num_batch_per_epoch is not None:
            world_size = get_world_size()
            num_samples = (
                num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
                world_size)
            dataset = _SlicedDataset(dataset, num_samples)

        # build sampler
        sampler_cfg = dataloader_cfg.pop('sampler')
        if isinstance(sampler_cfg, dict):
            sampler_seed = None if diff_rank_seed else seed
            sampler = DATA_SAMPLERS.build(
                sampler_cfg,
                default_args=dict(dataset=dataset, seed=sampler_seed))
        else:
            # fallback to raise error in dataloader
            # if `sampler_cfg` is not a valid type
            sampler = sampler_cfg

        # build batch sampler
        batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
        if batch_sampler_cfg is None:
            batch_sampler = None
        elif isinstance(batch_sampler_cfg, dict):
            batch_sampler = DATA_SAMPLERS.build(
                batch_sampler_cfg,
                default_args=dict(
                    sampler=sampler,
                    batch_size=dataloader_cfg.pop('batch_size')))
        else:
            # fallback to raise error in dataloader
            # if `batch_sampler_cfg` is not a valid type
            batch_sampler = batch_sampler_cfg

        # build dataloader
        init_fn: Optional[partial]

        if 'worker_init_fn' in dataloader_cfg:
            worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
            worker_init_fn_type = worker_init_fn_cfg.pop('type')
            if isinstance(worker_init_fn_type, str):
                worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
            elif callable(worker_init_fn_type):
                worker_init_fn = worker_init_fn_type
            else:
                raise TypeError(
                    'type of worker_init_fn should be string or callable '
                    f'object, but got {type(worker_init_fn_type)}')
            assert callable(worker_init_fn)
            init_fn = partial(worker_init_fn,
                              **worker_init_fn_cfg)  # type: ignore
        else:
            if seed is not None:
                disable_subprocess_warning = dataloader_cfg.pop(
                    'disable_subprocess_warning', False)
                assert isinstance(disable_subprocess_warning, bool), (
                    'disable_subprocess_warning should be a bool, but got '
                    f'{type(disable_subprocess_warning)}')
                init_fn = partial(
                    default_worker_init_fn,
                    num_workers=dataloader_cfg.get('num_workers'),
                    rank=get_rank(),
                    seed=seed,
                    disable_subprocess_warning=disable_subprocess_warning)
            else:
                init_fn = None

        # `persistent_workers` requires pytorch version >= 1.7
        if ('persistent_workers' in dataloader_cfg
                and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
            print_log(
                '`persistent_workers` is only available when '
                'pytorch version >= 1.7',
                logger='current',
                level=logging.WARNING)
            dataloader_cfg.pop('persistent_workers')

        # The default behavior of `collat_fn` in dataloader is to
        # merge a list of samples to form a mini-batch of Tensor(s).
        # However, in mmengine, if `collate_fn` is not defined in
        # dataloader_cfg, `pseudo_collate` will only convert the list of
        # samples into a dict without stacking the batch tensor.
        collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                        dict(type='pseudo_collate'))
        data_loader = DataLoader(
            dataset=dataset,
            sampler=sampler if batch_sampler is None else None,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=init_fn,
            **dataloader_cfg)
        return data_loader