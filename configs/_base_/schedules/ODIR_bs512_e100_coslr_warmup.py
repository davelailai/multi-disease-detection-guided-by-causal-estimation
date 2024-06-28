# optimizer
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# learning policy
# param_scheduler = [
#     # warm up learning rate scheduler
#    dict(
#         type='LinearLR',
#         start_factor=0.25,
#         by_epoch=True,
#         begin=0,
#         # about 2500 iterations for ImageNet-1k
#         end=10,
#         # update by iter
#         convert_to_iter_based=True),
#     # main learning rate scheduler
#     dict(
#         type='CosineAnnealingLR',
#         T_max=90,
#         by_epoch=True,
#         begin=10,
#         end=100,
#     )
# ]
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
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=1e-4))

# param_scheduler = [
#     dict(type='CosineAnnealingLR', T_max=100, by_epoch=True, begin=0, end=100)
# ]
# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
# auto_scale_lr = dict(base_batch_size=256)
