# Copyright (c) OpenMMLab. All rights reserved.
from .beitv1_head import BEiTV1Head
from .beitv2_head import BEiTV2Head
from .cae_head import CAEHead
from .cls_head import ClsHead
from .conformer_head import ConformerHead
from .contrastive_head import ContrastiveHead
from .deit_head import DeiTClsHead
from .efficientformer_head import EfficientFormerClsHead
from .grounding_head import GroundingHead
from .itc_head import ITCHead
from .itm_head import ITMHead
from .itpn_clip_head import iTPNClipHead
from .latent_heads import LatentCrossCorrelationHead, LatentPredictHead
from .levit_head import LeViTClsHead
from .linear_head import LinearClsHead
from .mae_head import MAEPretrainHead
from .margin_head import ArcFaceClsHead
from .mim_head import MIMHead
from .mixmim_head import MixMIMPretrainHead
from .mocov3_head import MoCoV3Head
from .multi_label_cls_head import MultiLabelClsHead
from .multi_label_csra_head import CSRAClsHead
from .multi_label_linear_head import MultiLabelLinearClsHead
from .multi_task_head import MultiTaskHead
from .seq_gen_head import SeqGenerationHead
from .simmim_head import SimMIMHead
from .spark_head import SparKPretrainHead
from .stacked_head import StackedLinearClsHead
from .swav_head import SwAVHead
from .vig_head import VigClsHead
from .vision_transformer_head import VisionTransformerClsHead
from .vqa_head import VQAGenerationHead
from .multi_label_MLDecoder_head import MLdecoderHead
from .multi_label_Q2L_head import Qeruy2Label
from .multi_label_GCNAdd_head import GCNClsHead
from .multi_label_GCN_head import GCNClsHead_base
from .multi_label_MLDecoder_Causal_head import MLCausaldecoderHead
from .multi_label_Q2L_head_new import Qeruy2Label_New
from .multi_label_MLDecoder_head_new import MLdecoderHead_New

__all__ = [
    'ClsHead',
    'LinearClsHead',
    'StackedLinearClsHead',
    'MultiLabelClsHead',
    'MultiLabelLinearClsHead',
    'VisionTransformerClsHead',
    'DeiTClsHead',
    'ConformerHead',
    'EfficientFormerClsHead',
    'ArcFaceClsHead',
    'CSRAClsHead',
    'MultiTaskHead',
    'LeViTClsHead',
    'VigClsHead',
    'BEiTV1Head',
    'BEiTV2Head',
    'CAEHead',
    'ContrastiveHead',
    'LatentCrossCorrelationHead',
    'LatentPredictHead',
    'MAEPretrainHead',
    'MixMIMPretrainHead',
    'SwAVHead',
    'MoCoV3Head',
    'MIMHead',
    'SimMIMHead',
    'SeqGenerationHead',
    'VQAGenerationHead',
    'ITCHead',
    'ITMHead',
    'GroundingHead',
    'iTPNClipHead',
    'SparKPretrainHead',
    'MLdecoderHead',
    'Qeruy2Label',
    'GCNClsHead',
    'GCNClsHead_base',
    'MLCausaldecoderHead',
    'Qeruy2Label_New',
    'MLdecoderHead_New'
]
