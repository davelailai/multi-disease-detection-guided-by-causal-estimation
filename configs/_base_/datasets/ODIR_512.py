dataset_type ='Ffa'
data_preprocessor = dict(
    # num_classes=7,
    # RGB format normalization parameters
    # mean=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
    # std=[0.26862954 * 255, 0.26130258 * 255, 0.27577711 * 255],
    mean=[0.5],
    std=[0.5],
    # convert image from BGR to RGB
    to_rgb=True,
)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',scale=512),
    dict(type='RandomCrop',crop_size=448),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',scale=512),
    dict(type='CenterCrop',crop_size=448),
    # dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='ResizeEdge',
#         scale=560,
#         edge='short',
#         backend='pillow',
#         interpolation='bicubic'),
#     dict(type='CenterCrop', crop_size=560),
#     dict(type='PackInputs'),
# ]
# folder='data/'
train_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/OIA-ODIR/Train.pkl',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/OIA-ODIR/onsite_test.pkl',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)


test_dataloader = dict(
    batch_size=64,
    num_workers=5,
    dataset=dict(
        type=dataset_type,
        # data_prefix=folder,
        ann_file='data/OIA-ODIR/offsite_test.pkl',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# If you want standard test, please manually configure the test dataset
val_evaluator = [
    dict(type='AverageAUC'),
    dict(type='AveragePrecision'), 
    dict(average=None, type='AverageAUC'),
  
    dict(type='MultiLabelMetric',
        items=['f1-score', 'precision', 'recall', 'support'],
        # average='both',
        thr=0.5),
    dict(type='MultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        average='micro',
        thr=0.5),
    dict(type='MultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        average=None,
        thr=0.5),
]
test_evaluator = val_evaluator