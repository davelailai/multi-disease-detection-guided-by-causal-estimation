# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn

from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .base import BaseClassifier
from typing import Dict, Optional, Tuple, Union

from mmengine.optim import OptimWrapper

@MODELS.register_module()
class ImageGanClassifier(BaseClassifier):
    """Image classifiers for supervised classification task.

    Args:
        backbone (dict): The backbone module. See
            :mod:`mmpretrain.models.backbones`.
        neck (dict, optional): The neck module to process features from
            backbone. See :mod:`mmpretrain.models.necks`. Defaults to None.
        head (dict, optional): The head module to do prediction and calculate
            loss from processed features. See :mod:`mmpretrain.models.heads`.
            Notice that if the head is not set, almost all methods cannot be
            used except :meth:`extract_feat`. Defaults to None.
        pretrained (str, optional): The pretrained checkpoint path, support
            local path and remote path. Defaults to None.
        train_cfg (dict, optional): The training setting. The acceptable
            fields are:

            - augments (List[dict]): The batch augmentation methods to use.
              More details can be found in
              :mod:`mmpretrain.model.utils.augment`.
            - probs (List[float], optional): The probability of every batch
              augmentation methods. If None, choose evenly. Defaults to None.

            Defaults to None.
        data_preprocessor (dict, optional): The config for preprocessing input
            data. If None or no specified type, it will use
            "ClsDataPreprocessor" as type. See :class:`ClsDataPreprocessor` for
            more details. Defaults to None.
        init_cfg (dict, optional): the config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: Optional[dict] = None,
                 head: Optional[dict] = None,
                 pretrained: Optional[str] = None,
                 train_cfg: Optional[dict] = None,
                 data_preprocessor: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        if pretrained is not None:
            init_cfg = dict(type='Pretrained', checkpoint=pretrained)

        data_preprocessor = data_preprocessor or {}

        if isinstance(data_preprocessor, dict):
            data_preprocessor.setdefault('type', 'ClsDataPreprocessor')
            data_preprocessor.setdefault('batch_augments', train_cfg)
            data_preprocessor = MODELS.build(data_preprocessor)
        elif not isinstance(data_preprocessor, nn.Module):
            raise TypeError('data_preprocessor should be a `dict` or '
                            f'`nn.Module` instance, but got '
                            f'{type(data_preprocessor)}')

        super(ImageGanClassifier, self).__init__(
            init_cfg=init_cfg, data_preprocessor=data_preprocessor)

        if not isinstance(backbone, nn.Module):
            backbone = MODELS.build(backbone)
        if neck is not None and not isinstance(neck, nn.Module):
            neck = MODELS.build(neck)
        if head is not None and not isinstance(head, nn.Module):
            head = MODELS.build(head)

        self.backbone = backbone
        self.neck = neck
        self.head = head

        # If the model needs to load pretrain weights from a third party,
        # the key can be modified with this hook
        if hasattr(self.backbone, '_checkpoint_filter'):
            self._register_load_state_dict_pre_hook(
                self.backbone._checkpoint_filter)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                mode: str = 'tensor'):
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor(s) without any
          post-processing, same as a common PyTorch Module.
        - "predict": Forward and return the predictions, which are fully
          processed to a list of :obj:`DataSample`.
        - "loss": Forward and return a dict of losses according to the given
          inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) self.set_requires_grad(self.head.DagmaMLP, False)
        base_optimizer_wrapper = optim_wrapper['backbone']
        base_optimizer_wrapper=optim_wrapper
        with base_optimizer_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
            parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
            base_optimizer_wrapper.update_params(parsed_losses)
        self.set_requires_grad(self.head.DagmaMLP, True)
 
        #Train causal matrix
        self.set_requires_grad(self.head, False)
        self.set_requires_grad(self.backbone, False)
        self.set_requires_grad(self.head.DagmaMLP, True)
        # causal_optimizer_wrapper = optim_wrapper['head']
        causal_optimizer_wrapper=optim_wrapper
        
        with causal_optimizer_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
            # losses.pop('loss')
            parsed_losses, log_vars_causal = self.parse_losses(losses)  # type: ignore
            causal_optimizer_wrapper.update_params(parsed_losses)
        self.set_requires_grad(self.head, True)
        self.set_requires_grad(self.backbone, True)


        return log_vars.update(log_vars) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. It's required if ``mode="loss"``.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of
              :obj:`mmpretrain.structures.DataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'tensor':
            feats = self.extract_feat(inputs)
            return self.head(feats) if self.with_head else feats
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}".')
    
    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        # 
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars
    
        # self.set_requires_grad(self.head.DagmaMLP, False)
        # # base_optimizer_wrapper = optim_wrapper['backbone']
        # base_optimizer_wrapper=optim_wrapper
        # with base_optimizer_wrapper.optim_context(self):
        #     data = self.data_preprocessor(data, True)
        #     losses = self._run_forward(data, mode='loss')  # type: ignore
        #     parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        #     base_optimizer_wrapper.update_params(losses['loss'])
        # self.set_requires_grad(self.head.DagmaMLP, True)
 
        # #Train causal matrix
        # self.set_requires_grad(self.head, False)
        # self.set_requires_grad(self.backbone, False)
        # self.set_requires_grad(self.head.DagmaMLP, True)
        # # causal_optimizer_wrapper = optim_wrapper['head']
        # causal_optimizer_wrapper=optim_wrapper
        
        # with causal_optimizer_wrapper.optim_context(self):
        #     data = self.data_preprocessor(data, True)
        #     losses = self._run_forward(data, mode='loss')  # type: ignore
        #     # losses.pop('loss')
        #     parsed_losses, log_vars_causal = self.parse_losses(losses)  # type: ignore
        #     causal_optimizer_wrapper.update_params(losses['causal_loss'])
        # self.set_requires_grad(self.head, True)
        # self.set_requires_grad(self.backbone, True)


        # return log_vars.update(log_vars)
    
    def set_requires_grad(sel, nets, requires_grad=False):
        """Set requires_grad for all the networks.

        Args:
            nets (nn.Module | list[nn.Module]): A list of networks or a single
                network.
            requires_grad (bool): Whether the networks require gradients or not.
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def extract_feat(self, inputs, stage='neck'):
        """Extract features from the input tensor with shape (N, C, ...).

        Args:
            inputs (Tensor): A batch of inputs. The shape of it should be
                ``(num_samples, num_channels, *img_shape)``.
            stage (str): Which stage to output the feature. Choose from:

                - "backbone": The output of backbone network. Returns a tuple
                  including multiple stages features.
                - "neck": The output of neck module. Returns a tuple including
                  multiple stages features.
                - "pre_logits": The feature before the final classification
                  linear layer. Usually returns a tensor.

                Defaults to "neck".

        Returns:
            tuple | Tensor: The output of specified stage.
            The output depends on detailed implementation. In general, the
            output of backbone and neck is a tuple and the output of
            pre_logits is a tensor.

        Examples:
            1. Backbone output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='backbone')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64, 56, 56])
            torch.Size([1, 128, 28, 28])
            torch.Size([1, 256, 14, 14])
            torch.Size([1, 512, 7, 7])

            2. Neck output

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/resnet/resnet18_8xb32_in1k.py').model
            >>> cfg.backbone.out_indices = (0, 1, 2, 3)  # Output multi-scale feature maps
            >>> model = build_classifier(cfg)
            >>>
            >>> outs = model.extract_feat(torch.rand(1, 3, 224, 224), stage='neck')
            >>> for out in outs:
            ...     print(out.shape)
            torch.Size([1, 64])
            torch.Size([1, 128])
            torch.Size([1, 256])
            torch.Size([1, 512])

            3. Pre-logits output (without the final linear classifier head)

            >>> import torch
            >>> from mmengine import Config
            >>> from mmpretrain.models import build_classifier
            >>>
            >>> cfg = Config.fromfile('configs/vision_transformer/vit-base-p16_pt-64xb64_in1k-224.py').model
            >>> model = build_classifier(cfg)
            >>>
            >>> out = model.extract_feat(torch.rand(1, 3, 224, 224), stage='pre_logits')
            >>> print(out.shape)  # The hidden dims in head is 3072
            torch.Size([1, 3072])
        """  # noqa: E501
        assert stage in ['backbone', 'neck', 'pre_logits'], \
            (f'Invalid output stage "{stage}", please choose from "backbone", '
             '"neck" and "pre_logits"')

        x = self.backbone(inputs)
        
        # import os
        # import pickle
        # save_dir='work_dirs_MICCAI_new/causal_matrix'
        # W = self.head.DagmaMLP.fc1_to_adj()
        # with open(os.path.join(save_dir,'FFA_ML_matrix_1.pkl'), 'wb') as f:
        #     pickle.dump(W, f)
        
        # import numpy as np
        # np.fill_diagonal(W, 0)
        # # label=['D', 'G', 'C', 'A', 'H', 'M', 'O']
        # label=['L', 'TP', 'ST', 'SH', 'NP', 'VA']
        
        # self.plot_confusion_matrix(W.T,label,save_dir=save_dir,epoch='FFA_ML_d30')
        # self.plot_confusion_matrix(W.T,label,save_dir=save_dir,topk=1,epoch='FFA_ML_d30')
        # self.plot_confusion_matrix(W.T,label,save_dir=save_dir,topk=2,epoch='FFA_ML_d30')
        # self.plot_confusion_matrix(W.T,label,save_dir=save_dir,topk=3,epoch='FFA_ML_d30')


        if stage == 'backbone':
            return x

        if self.with_neck:
            x = self.neck(x)
        if stage == 'neck':
            return x

        assert self.with_head and hasattr(self.head, 'pre_logits'), \
            "No head or the head doesn't implement `pre_logits` method."
        return self.head.pre_logits(x)

    def loss(self, inputs: torch.Tensor,
             data_samples: List[DataSample]) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample]): The annotation data of
                every samples.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        feats = self.extract_feat(inputs)
        return self.head.loss(feats, data_samples)

    def predict(self,
                inputs: torch.Tensor,
                data_samples: Optional[List[DataSample]] = None,
                **kwargs) -> List[DataSample]:
        """Predict results from a batch of inputs.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (List[DataSample], optional): The annotation
                data of every samples. Defaults to None.
            **kwargs: Other keyword arguments accepted by the ``predict``
                method of :attr:`head`.
        """
        feats = self.extract_feat(inputs)
        return self.head.predict(feats, data_samples, **kwargs)

    def get_layer_depth(self, param_name: str):
        """Get the layer-wise depth of a parameter.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            Tuple[int, int]: The layer-wise depth and the max depth.
        """
        if hasattr(self.backbone, 'get_layer_depth'):
            return self.backbone.get_layer_depth(param_name, 'backbone.')
        else:
            raise NotImplementedError(
                f"The backbone {type(self.backbone)} doesn't "
                'support `get_layer_depth` by now.')
    def plot_confusion_matrix(self,
                            confusion_matrix,
                            labels,
                            save_dir=None,
                            topk=False,
                            epoch=None,
                            show=True,
                            title='Normalized Confusion Matrix',
                            color_theme='plasma'):
        """Draw confusion matrix with matplotlib.

        Args:
            confusion_matrix (ndarray): The confusion matrix.
            labels (list[str]): List of class names.
            save_dir (str|optional): If set, save the confusion matrix plot to the
                given path. Default: None.
            show (bool): Whether to show the plot. Default: True.
            title (str): Title of the plot. Default: `Normalized Confusion Matrix`.
            color_theme (str): Theme of the matrix color map. Default: `plasma`.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator
        import os
        # normalize the confusion matrix
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        confusion_matrix = \
            confusion_matrix.astype(np.float32) / per_label_sums * 100
        if topk:
            # Find the indices of the top-k values in each row
            top_k_indices = np.argsort(confusion_matrix, axis=1)[:, -topk:]

            # Create a binary matrix where only the top-k values are set to 1
            binary_matrix = np.zeros_like(confusion_matrix)
            rows = np.arange(binary_matrix.shape[0])[:, np.newaxis]
            binary_matrix[rows, top_k_indices] = 1
            confusion_matrix=binary_matrix
            color_theme='Greys'
            epoch=epoch+'_top_'+str(topk)

        num_classes = len(labels)
        fig, ax = plt.subplots(
            figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=300)
        cmap = plt.get_cmap(color_theme)
        im = ax.imshow(confusion_matrix, cmap=cmap)
        plt.colorbar(mappable=im, ax=ax)

        title_font = {'weight': 'bold', 'size': 3}
        # ax.set_title(title, fontdict=title_font)
        label_font = {'size': 3}
        # plt.ylabel('Ground Truth Label', fontdict=label_font)
        # plt.xlabel('Prediction Label', fontdict=label_font)

        # draw locator
        xmajor_locator = MultipleLocator(1)
        xminor_locator = MultipleLocator(0.5)
        ax.xaxis.set_major_locator(xmajor_locator)
        ax.xaxis.set_minor_locator(xminor_locator)
        ymajor_locator = MultipleLocator(1)
        yminor_locator = MultipleLocator(0.5)
        ax.yaxis.set_major_locator(ymajor_locator)
        ax.yaxis.set_minor_locator(yminor_locator)

        # draw grid
        ax.grid(True, which='minor', linestyle='-')

        # draw label
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_classes))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        ax.tick_params(
            axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
        plt.setp(
            ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

        # # draw confution matrix value
        # for i in range(num_classes):
        #     for j in range(num_classes):
        #         ax.text(
        #             j,
        #             i,
        #             '{}%'.format(
        #                 int(confusion_matrix[
        #                     i,
        #                     j]) if not np.isnan(confusion_matrix[i, j]) else -1),
        #             ha='center',
        #             va='center',
        #             color='w',
        #             size=7)

        ax.set_ylim(len(confusion_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

        fig.tight_layout()
        if save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, epoch+'causal_matrix.png'), format='png')
        # if show:
        #     plt.show()
