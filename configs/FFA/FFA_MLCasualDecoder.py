_base_ = [
    # '../_base_/models/resnet101.py',
    './backbone.py',
    '../_base_/datasets/ffa_560.py',
    '../_base_/schedules/FFA_bs512_100e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs_MICCAI_new/FFA6_new_set/resnet50_448/MLCausalDecoder_red_direct_relation_d10_lr5e-3_head1_e100'

# optim_wrapper = dict(backbone=dict(type='OptimWrapper', optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)),
#                     head=dict(type='OptimWrapper',optimizer=dict(type='SGD', lr=1e-2, momentum=0.9, weight_decay=0.0001)),
#                     constructor='MultiOptimWrapperConstructor')

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10)
        })
        )



model = dict(
    type='ImageGanClassifier',
    neck=None,
    head=dict(
        type='MLdecoderHead_New',
        # num_classes=6,
        # decoder_embedding=768,
        # in_channels=1280,
        causal=True,
        d2=10,
        feature_reg=True,
        causal_reg=True,
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True),    
        # s=1.0, 
        # lambda1=1e-1, 
        # m=0.1,
        weight=0.1,
       ),
       )

# train_dataloader = dict(
#     batch_size=4)

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

    
default_hooks = dict(
    # record the time of every iteration.
    logger=dict(type='LoggerHook', interval=50),

    checkpoint=dict(type='CheckpointHook', interval=20,save_best='auto'),
)