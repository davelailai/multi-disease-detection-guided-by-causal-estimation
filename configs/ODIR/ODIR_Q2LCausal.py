_base_ = [
    # '../_base_/models/efficientnet_v2/efficientnetv2_s.py',
    './backbone.py',
    '../_base_/datasets/ODIR_512.py',
    '../_base_/schedules/ODIR_bs512_e100_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs_MICCAI_new/ODIR_7/resnet50_448/Q2L_causal_relation_direct_d30_lr5e-3_head10_e100'

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10)
        }))

model = dict(
    type='ImageGanClassifier',
    # backbone=dict(
    #     type='EfficientNetV2', 
    #     arch='s',
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth',
    #         prefix='backbone')),
    head=dict(
        type='Qeruy2Label_New',
        # num_class=8,
        # num_channels=1280,
        position_embedding='sine',
        enc_layers=1,
        dec_layers=2,
        dim_feedforward=256,
        hidden_dim=128,
        nheads=4,
        img_size=448,
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True),
        causal=True,
        d2=30,
        feature_reg=True,
        causal_reg=True,
        # causal_num=20,
        # s=1.0, 
        # lambda1=1e-1, 
        # m=0.1,
        weight=0.1,
        ),
    neck=None,
    # init_cfg=dict(type='Pretrained', checkpoint='work_dirs_MICCAI/ODIR_8/resnet50_448/Q2L_causal_lr5e-3_e100/epoch_60.pth')
    )

# train_dataloader = dict(
#     batch_size=4)
# If you want standard test, please manually configure the test dataset
# val_evaluator = [
#     dict(type='AveragePrecision'),
#     dict(type='AverageAUC'),
#     dict(average=None, type='AverageAUC'),
#     dict(type='MultiLabelMetric',
#         items=['f1-score','precision', 'recall', 'f1-score','support'],
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

custom_hooks=[
    dict(type='WarmupParamHook', param_name='mu', dec_name='mu_dec',module_name='head', warmup_epochs=100)
    ]
