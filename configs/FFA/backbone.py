## EfficientnetV2
# model = dict(
#     type='ImageGanClassifier',
#     backbone=dict(
#         type='EfficientNetV2', 
#         arch='s',
#         init_cfg=dict(
#             type='Pretrained',
#             checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth',
#             prefix='backbone')),
#     head=dict(
#         num_classes=6,
#         in_channels=1280,
#         ))

## resnet50
# model = dict(
#     type='ImageGanClassifier',
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(3, ),
#         style='pytorch',
#         init_cfg=dict(
#             checkpoint=
#             'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
#             prefix='backbone',
#             type='Pretrained'),),
#     head=dict(
#         num_classes=6,
#         in_channels=2048,
#         ))

## resnet101
model = dict(
    type='ImageGanClassifier',
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
            prefix='backbone',
            type='Pretrained'),
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        num_classes=6,
        in_channels=2048,
        ))
## MAE
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='VisionTransformer',
#         arch='large',
#         img_size=224,
#         patch_size=16,
#         # frozen_stages=24,
#         out_type='cls_token',
#         final_norm=True,
#         init_cfg=dict(type='Pretrained', 
#                     checkpoint='/users/lailai/sharedscratch/openmmlab/mmpretrain/mmpretrain_RETFound_cfp_weights.pth',
#         #             # checkpoint='https://download.openmmlab.com/mmselfsup/1.x/mae/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k/mae_vit-large-p16_8xb512-fp16-coslr-1600e_in1k_20220825-cc7e98c9.pth',
#                     prefix='backbone'
#                     )
#         # init_cfg=dict(type='Pretrained', checkpoint='', prefix='backbone.')
#         ),
#     neck=dict(type='ClsBatchNormNeck', input_features=1024),
#     head=dict(
#         # type='VisionTransformerClsHead',
#         num_classes=6,
#         in_channels=1024,
#         # loss=dict(type='CrossEntropyLoss',class_weight=[1,2]),
#         # init_cfg=[dict(type='TruncNormal', layer='Linear', std=0.01)]
#         ))