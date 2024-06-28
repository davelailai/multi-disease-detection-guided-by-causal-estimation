# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/Kevinz-code/CSRA
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.modules.transformer import _get_activation_fn
from torch import Tensor
from typing import Optional
from mmengine.model import BaseModule, ModuleList
from typing import Dict, List, Optional, Tuple

from mmpretrain.registry import MODELS
from .multi_label_cls_head import MultiLabelClsHead
from mmpretrain.structures import DataSample
from ..utils import *


@MODELS.register_module()
class MLdecoderHead(MultiLabelClsHead):
    """Class-specific residual attention classifier head.

    Please refer to the `Residual Attention: ML-Decoder: Scalable and Versatile Classification Head https://arxiv.org/abs/2111.12933`_
    for details.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        num_heads (int): Number of residual at tensor heads.
        loss (dict): Config of classification loss.
        lam (float): Lambda that combines global average and max pooling
            scores.
        init_cfg (dict, optional): The extra init config of layers.
            Defaults to use ``dict(type='Normal', layer='Linear', std=0.01)``.
    """
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                initial_num_features=2048, zsl=0, causal=False,s=1.0, lambda1=0.02, m=0.1, weight=1e-2, **kwargs):
        super(MLdecoderHead, self).__init__(**kwargs)

        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

         # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            # query_embed.requires_grad_(False)
        else:
            query_embed = None

        # self.DagmaMLP= DagmaMLP(dims=[decoder_embedding, 10, 1], bias=True, dtype=query_embed.type())
        
        self.causal = causal
        if self.causal:
            self.DagmaMLP= DagmaMLP(dims=[decoder_embedding, 10, 1], bias=True,dtype=torch.float)
            self.s = s
            self.lambda1=lambda1
            self.mu=m
            self.w = weight
     
        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None
        self.logsoftmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, x):
        x = self.pre_logits(x)
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        return logits
    
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
                Multiple stage inputs are acceptable but only the last stage
                will be used to classify. The shape of every item should be
                ``(num_samples, num_classes)``.
            data_samples (List[DataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # The part can be traced by torch.fx
        cls_score = self(feats)

        # The part can not be traced by torch.fx
        losses = self._get_loss(cls_score, data_samples, **kwargs)
        if self.causal:
            causal_loss = self.causal_estimation()
            losses['causal_loss']=causal_loss
        return losses

    def log_mse_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the logarithm of the MSE loss:
            .. math::
                \frac{d}{2} \log\left( \frac{1}{n} \sum_{i=1}^n (\mathrm{output}_i - \mathrm{target}_i)^2 \right)
        
        Parameters
        ----------
        output : torch.Tensor
            :math:`(n,d)` output of the model
        target : torch.Tensor
            :math:`(n,d)` input dataset

        Returns
        -------
        torch.Tensor
            A scalar value of the loss.
        """
        n, d = target.shape
        loss = 0.5 * d * torch.log(1 / n * torch.sum((output - target) ** 2))
        return loss
    
    def causal_estimation(self):
        h_val = self.DagmaMLP.h_func(self.s)
        X_hat = self.DagmaMLP(self.decoder.query_embed.weight)
        score = self.log_mse_loss(X_hat, self.decoder.query_embed.weight)
        l1_reg = self.lambda1 * self.DagmaMLP.fc1_l1_reg()
        obj = self.mu * (score + l1_reg) + h_val
        return self.w* obj



@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt
