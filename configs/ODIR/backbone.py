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
#         num_classes=8,
#         in_channels=1280,
#         ))

## resnet50
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
            type='Pretrained'),),
    head=dict(
        num_classes=7,
        in_channels=2048,
        ))

## resnet101
# model = dict(
#     type='ImageGanClassifier',
#     backbone=dict(
#         depth=101,
#         init_cfg=dict(
#             checkpoint=
#             'https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth',
#             prefix='backbone',
#             type='Pretrained'),
#         num_stages=4,
#         out_indices=(3, ),
#         style='pytorch',
#         type='ResNet'),
#     head=dict(
#         num_classes=8,
#         in_channels=2048,
#         ))