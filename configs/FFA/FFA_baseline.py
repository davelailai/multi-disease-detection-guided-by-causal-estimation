_base_ = [
    # '../_base_/models/resnet101.py',
    './backbone.py',
    '../_base_/datasets/ffa_560.py',
    '../_base_/schedules/FFA_bs512_100e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs_TMI_test/FFA6/MAE_224/base_lr5e-3_e100'

model = dict(
    # neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        # num_classes=6,
        # in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        # in_channels=2048,
       ))



# If you want standard test, please manually configure the test dataset
# val_evaluator = [
#     dict(type='AveragePrecision'),
#     dict(type='AverageAUC'),
#     dict(average=None, type='AverageAUC'),
#     dict(type='MultiLabelMetric',
#         items=['f1-score', 'precision', 'recall', 'support'],
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

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

default_hooks = dict(
    # record the time of every iteration.
    logger=dict(type='LoggerHook', interval=50),

    checkpoint=dict(type='CheckpointHook', interval=20,save_best='auto'),
)
