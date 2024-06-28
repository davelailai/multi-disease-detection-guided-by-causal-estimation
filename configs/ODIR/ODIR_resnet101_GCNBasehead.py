_base_ = [
    # '../_base_/models/resnet101.py',
    './backbone.py',
    '../_base_/datasets/ODIR_512.py',
    '../_base_/schedules/FFA_bs512_100e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs_MICCAI_new/ODIR_7_new_set/resnet50_448/GCNBasehead_QL_causal_lr5e-3_e100'

model = dict(
    type='ImageClassifier',
    neck=None,
    head=dict(
        type='GCNClsHead_base',
        label_file='/users/lailai/sharedscratch/openmmlab/mmpretrain/work_dirs_MICCAI_new/causal_matrix/OD_QL_matrix.pkl',
        name='OIA-ODIR',
        t=0,
        itself=False,
        # num_classes=6,
        in_channels=8,
        inter_channels=256,
        out_channels=2048,
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True),
       ))

param_scheduler = [
    dict(
        # begin=0,
        by_epoch=True,
        div_factor=1e2,
        total_steps=100,
        pct_start=0.1,
        # end=100,
        eta_max=5e-3,
        final_div_factor=1e3,
        type='OneCycleLR'),
]

# If you want standard test, please manually configure the test dataset
# val_evaluator = [
#     dict(type='AveragePrecision'),
#     dict(type='AverageAUC'),
#     dict(average=None, type='AverageAUC'),
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

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True,
        # detect_anomalous_params=True
        )

default_hooks = dict(
    # record the time of every iteration.
    logger=dict(type='LoggerHook', interval=50),

    checkpoint=dict(type='CheckpointHook', interval=20,save_best='auto'),
)