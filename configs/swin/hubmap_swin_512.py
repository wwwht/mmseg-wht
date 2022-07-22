_base_ = [
   '../_base_/models/upernet_swin.py', '../_base_/datasets/my_hubmap_512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True),
    decode_head=dict(
        in_channels=[96, 192, 384, 768], 
        num_classes=2,
        loss_decode=dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(in_channels=384, num_classes=2))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=2000)
evaluation = dict(interval=100, metric='mIoU', pre_eval=True)
data = dict(samples_per_gpu=16)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='MMSegWandbHook',
             init_kwargs={
                 'entity': 'wwwht',
                 'project': 'mmseg-test'
             },
             log_checkpoint=True,
             log_checkpoint_metadata=True,
             num_eval_images=100)
    ]) 
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whether count by epoch or not.
    interval=1000)  # The save interval.
