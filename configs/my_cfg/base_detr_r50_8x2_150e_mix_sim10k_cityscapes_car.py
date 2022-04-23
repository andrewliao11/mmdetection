import os
import wandb

_base_ = [
    #'../_base_/datasets/cityscapes_detection.py', 
    '../_base_/default_runtime.py'
]


model = dict(
    type='DETR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='DETRHead',
        #num_classes=80,
        num_classes=8,
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=dict(
                        type='MultiheadAttention',
                        embed_dims=256,
                        num_heads=8,
                        dropout=0.1),
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        loss_cls=dict(
            type='CrossEntropyLoss',
            bg_cls_weight=0.1,
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=1.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=100))

# dataset settings

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


sim10k_data_root = os.environ["HOME"] + '/datasets/sim10k/'
sim10k_dataset = dict(
    type='RepeatDataset',
    times=8,
    dataset=dict(
        type='CityscapesDataset',
        ann_file=sim10k_data_root +
        'annotations/voc2012_annotations.json',
    img_prefix=sim10k_data_root + 'JPEGImages/',
    pipeline=train_pipeline)
)


cityscapes_data_root = os.environ["HOME"] + '/datasets/cityscapes_car/'
cityscapes_trian_dataset = dict(
    type='RepeatDataset',
    times=8,
    dataset=dict(
        type='CityscapesDataset',
        ann_file=cityscapes_data_root +
        'annotations/instancesonly_filtered_gtFine_train.json',
    img_prefix=cityscapes_data_root + 'leftImg8bit/train/',
    pipeline=train_pipeline)
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='ConcatDataset',
        datasets=[sim10k_dataset, cityscapes_trian_dataset],
        separate_eval=False),
    val=dict(
        type='CityscapesDataset',
        ann_file=cityscapes_data_root +
        'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=cityscapes_data_root + 'leftImg8bit/val/',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        ann_file=cityscapes_data_root +
        'annotations/instancesonly_filtered_gtFine_test.json',
        img_prefix=cityscapes_data_root + 'leftImg8bit/test/',
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox')
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={'backbone': dict(lr_mult=0.1, decay_mult=1.0)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[100])
runner = dict(type='EpochBasedRunner', max_epochs=150)


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
])

# wandb log
#custom_imports = dict(imports=['mmdet.core.hook.wandblogger_hook'], allow_failed_imports=False)
custom_hooks = [
    dict(type='WandbLogger',
        wandb_init_kwargs={
            'entity': "andrew-liao",
            'project': "label-translation-detr-sim10k_cityscapes_car", 
            'name': "test"
            },
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)
]
checkpoint_config = dict(interval=10)