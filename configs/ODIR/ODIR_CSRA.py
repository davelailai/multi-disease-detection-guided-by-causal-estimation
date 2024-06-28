_base_ = [
    # '../_base_/models/efficientnet_v2/efficientnetv2_s.py',
    './backbone.py',
    '../_base_/datasets/ODIR_512.py',
    '../_base_/schedules/ODIR_bs512_e100_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs_MICCAI_new/ODIR_7/resnet50_448/CSRA_lr5e-3_e100'
model = dict(
    head=dict(
        type='CSRAClsHead',
        # num_classes=8,
        num_heads=1,
        # in_channels=1280,
        lam=0.1,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0)),
    neck=None)


# If you want standard test, please manually configure the test dataset
# val_evaluator = [
#     dict(average=None, type='AverageAUC'),
#     dict(type='AveragePrecision'),
#     dict(type='AverageAUC'),
#     dict(type='MultiLabelMetric',
#         items=['f1-score','precision', 'recall', 'support'],
#         # average='both',
#         thr=0.5),
#     dict(type='MultiLabelMetric',
#         items=['precision', 'recall', 'f1-score','support'],
#         average='micro',
#         thr=0.5),
#     dict(type='MultiLabelMetric',
#         items=['precision', 'recall', 'f1-score','support'],
#         average=None,
#         thr=0.5),
# ]
# test_evaluator = val_evaluator

# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    # record the time of every iteration.
    logger=dict(type='LoggerHook', interval=50),

    checkpoint=dict(type='CheckpointHook', interval=20,save_best='auto'),
)
