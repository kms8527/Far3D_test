checkpoint_config = dict(interval=28130, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [
    dict(type='UseGtDepthHook', stop_gt_depth_iter=22000),
    dict(type='PeriodicCkptHook', interval=22000, save_dir='ckpts')
]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/home/a/opensource/Far3D_test/work_dirs/far3d_nuscene'
load_from = 'ckpts/fcos3d_vovnet_imgbackbone-remapped.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
backbone_norm_cfg = dict(type='LN', requires_grad=True)
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
vehicle_class_names = ['car', 'truck', 'construction_vehicle', 'bus']
vehicle_class_ids = [0, 1, 2, 3]
num_gpus = 1
batch_size = 1
num_iters_per_epoch = 28130
num_epochs = 24
embed_dims = 256
queue_length = 1
num_frame_losses = 1
collect_keys = [
    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp',
    'ego_pose', 'ego_pose_inv'
]
depthnet_config = dict(
    type=0,
    hidden_dim=256,
    num_depth_bins=50,
    depth_min=0.1,
    depth_max=110,
    stride=8)
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)
model = dict(
    type='Far3D',
    use_grid_mask=True,
    stride=[8, 16, 32, 64],
    position_level=[0, 1, 2, 3],
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage2', 'stage3', 'stage4', 'stage5')),
    img_neck=dict(
        type='FPN',
        start_level=1,
        add_extra_convs='on_output',
        relu_before_extra_convs=True,
        in_channels=[256, 512, 768, 1024],
        out_channels=256,
        num_outs=4),
    img_roi_head=dict(
        type='YOLOXHeadCustom',
        num_classes=10,
        in_channels=256,
        strides=[8, 16, 32, 64],
        train_cfg=dict(
            assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
        test_cfg=dict(
            score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
        pred_with_depth=True,
        depthnet_config=dict(
            type=0,
            hidden_dim=256,
            num_depth_bins=50,
            depth_min=0.1,
            depth_max=110,
            stride=8),
        reg_depth_level='p3',
        pred_depth_var=False,
        loss_depth2d=dict(type='L1Loss', loss_weight=1.0),
        sample_with_score=True,
        threshold_score=0.1,
        topk_proposal=None,
        return_context_feat=True),
    pts_bbox_head=dict(
        type='FarHead',
        num_classes=10,
        in_channels=256,
        num_query=644,
        memory_len=1024,
        topk_proposals=256,
        num_propagated=256,
        scalar=10,
        noise_scale=1.0,
        dn_weight=1.0,
        split=0.75,
        offset=0.5,
        offset_p=0.0,
        num_smp_per_gt=3,
        with_dn=True,
        with_ego_pos=True,
        use_vehicle_kinematics=True,
        vehicle_class_ids=[0, 1, 2, 3],
        debug_vehicle_kinematics=False,
        debug_vehicle_kinematics_max=64,
        add_query_from_2d=True,
        pred_box_var=False,
        depthnet_config=dict(
            type=0,
            hidden_dim=256,
            num_depth_bins=50,
            depth_min=0.1,
            depth_max=110,
            stride=8),
        train_use_gt_depth=True,
        add_multi_depth_proposal=True,
        multi_depth_config=dict(topk=1, range_min=30),
        return_bbox2d_scores=True,
        return_context_feat=True,
        code_size=8,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='Detr3DTransformer',
            decoder=dict(
                type='Detr3DTransformerDecoder',
                embed_dims=256,
                num_layers=6,
                transformerlayers=dict(
                    type='Detr3DTemporalDecoderLayer',
                    batch_first=True,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='DeformableFeatureAggregationCuda',
                            embed_dims=256,
                            num_groups=8,
                            num_levels=4,
                            num_cams=6,
                            dropout=0.1,
                            num_pts=13,
                            bias=2.0)
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    with_cp=True,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=[0.2, 0.2, 8],
            point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))))
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
ida_aug_conf = dict(
    resize_lim=(0.47, 0.55),
    final_dim=(640, 960),
    final_dim_f=(640, 720),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    rand_flip=False)
train_pipeline = [
    dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox=True,
        with_label=True,
        with_bbox_depth=True),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='NuScenesResizeCropFlipRotImageV2',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.55),
            final_dim=(640, 960),
            final_dim_f=(640, 720),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            rand_flip=False)),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='NuScenesPadMultiViewImage', size='same2max'),
    dict(
        type='NuScenesDownsampleQuantizeInstanceDepthmap',
        downsample=8,
        depth_config=dict(
            type=0,
            hidden_dim=256,
            num_depth_bins=50,
            depth_min=0.1,
            depth_max=110,
            stride=8)),
    dict(
        type='PETRFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes', 'gt_labels',
            'centers2d', 'depths', 'prev_exists', 'lidar2img', 'intrinsics',
            'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose',
            'ego_pose_inv'
        ],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                   'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d',
                   'img_norm_cfg', 'scene_token', 'gt_bboxes_3d',
                   'gt_labels_3d', 'ins_depthmap', 'ins_depthmap_mask'))
]
test_pipeline = [
    dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='NuScenesResizeCropFlipRotImageV2',
        data_aug_conf=dict(
            resize_lim=(0.47, 0.55),
            final_dim=(640, 960),
            final_dim_f=(640, 720),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            rand_flip=False)),
    dict(
        type='NormalizeMultiviewImage',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='NuScenesPadMultiViewImage', size='same2max'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='PETRFormatBundle3D',
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'img', 'lidar2img', 'intrinsics', 'extrinsics',
                    'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token'))
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type='CustomNuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_train.pkl',
        with_velocity=False,
        load_interval=1,
        num_frame_losses=1,
        seq_split_num=2,
        seq_mode=True,
        pipeline=[
            dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=True,
                with_label_3d=True,
                with_bbox=True,
                with_label=True,
                with_bbox_depth=True),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='NuScenesResizeCropFlipRotImageV2',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.55),
                    final_dim=(640, 960),
                    final_dim_f=(640, 720),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    rand_flip=False)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='NuScenesPadMultiViewImage', size='same2max'),
            dict(
                type='NuScenesDownsampleQuantizeInstanceDepthmap',
                downsample=8,
                depth_config=dict(
                    type=0,
                    hidden_dim=256,
                    num_depth_bins=50,
                    depth_min=0.1,
                    depth_max=110,
                    stride=8)),
            dict(
                type='PETRFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                collect_keys=[
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'gt_bboxes_3d', 'gt_labels_3d', 'img', 'gt_bboxes',
                    'gt_labels', 'centers2d', 'depths', 'prev_exists',
                    'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                    'img_timestamp', 'ego_pose', 'ego_pose_inv'
                ],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'box_mode_3d',
                           'box_type_3d', 'img_norm_cfg', 'scene_token',
                           'gt_bboxes_3d', 'gt_labels_3d', 'ins_depthmap',
                           'ins_depthmap_mask'))
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'prev_exists',
            'img_metas'
        ],
        queue_length=1,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type='CustomNuScenesDataset',
        pipeline=[
            dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NuScenesResizeCropFlipRotImageV2',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.55),
                    final_dim=(640, 960),
                    final_dim_f=(640, 720),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    rand_flip=False)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='NuScenesPadMultiViewImage', size='same2max'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token'))
                ])
        ],
        data_root='data/nuscenes/',
        with_velocity=False,
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'img_metas'
        ],
        queue_length=1,
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        load_interval=1,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        use_valid_flag=True),
    test=dict(
        type='CustomNuScenesDataset',
        pipeline=[
            dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
            dict(
                type='NuScenesResizeCropFlipRotImageV2',
                data_aug_conf=dict(
                    resize_lim=(0.47, 0.55),
                    final_dim=(640, 960),
                    final_dim_f=(640, 720),
                    bot_pct_lim=(0.0, 0.0),
                    rot_lim=(0.0, 0.0),
                    rand_flip=False)),
            dict(
                type='NormalizeMultiviewImage',
                mean=[103.53, 116.28, 123.675],
                std=[57.375, 57.12, 58.395],
                to_rgb=False),
            dict(type='NuScenesPadMultiViewImage', size='same2max'),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='PETRFormatBundle3D',
                        collect_keys=[
                            'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'img', 'lidar2img', 'intrinsics', 'extrinsics',
                            'timestamp', 'img_timestamp', 'ego_pose',
                            'ego_pose_inv'
                        ],
                        meta_keys=('filename', 'ori_shape', 'img_shape',
                                   'pad_shape', 'scale_factor', 'flip',
                                   'box_mode_3d', 'box_type_3d',
                                   'img_norm_cfg', 'scene_token'))
                ])
        ],
        data_root='data/nuscenes/',
        with_velocity=False,
        collect_keys=[
            'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
            'img_timestamp', 'ego_pose', 'ego_pose_inv', 'img', 'img_metas'
        ],
        queue_length=1,
        ann_file='data/nuscenes/nuscenes2d_temporal_infos_val.pkl',
        load_interval=1,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True),
        use_valid_flag=True),
    shuffler_sampler=dict(type='InfiniteGroupEachSampleInBatchSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))
optimizer = dict(
    type='AdamW',
    lr=0.0002,
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))),
    weight_decay=0.01)
optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    min_lr_ratio=0.001)
evaluation = dict(
    interval=675120,
    pipeline=[
        dict(type='NuScenesLoadMultiViewImageFromFiles', to_float32=True),
        dict(
            type='NuScenesResizeCropFlipRotImageV2',
            data_aug_conf=dict(
                resize_lim=(0.47, 0.55),
                final_dim=(640, 960),
                final_dim_f=(640, 720),
                bot_pct_lim=(0.0, 0.0),
                rot_lim=(0.0, 0.0),
                rand_flip=False)),
        dict(
            type='NormalizeMultiviewImage',
            mean=[103.53, 116.28, 123.675],
            std=[57.375, 57.12, 58.395],
            to_rgb=False),
        dict(type='NuScenesPadMultiViewImage', size='same2max'),
        dict(
            type='MultiScaleFlipAug3D',
            img_scale=(1333, 800),
            pts_scale_ratio=1,
            flip=False,
            transforms=[
                dict(
                    type='PETRFormatBundle3D',
                    collect_keys=[
                        'lidar2img', 'intrinsics', 'extrinsics', 'timestamp',
                        'img_timestamp', 'ego_pose', 'ego_pose_inv'
                    ],
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'img', 'lidar2img', 'intrinsics', 'extrinsics',
                        'timestamp', 'img_timestamp', 'ego_pose',
                        'ego_pose_inv'
                    ],
                    meta_keys=('filename', 'ori_shape', 'img_shape',
                               'pad_shape', 'scale_factor', 'flip',
                               'box_mode_3d', 'box_type_3d', 'img_norm_cfg',
                               'scene_token'))
            ])
    ])
find_unused_parameters = False
runner = dict(type='IterBasedRunner', max_iters=675120)
gpu_ids = range(0, 1)
