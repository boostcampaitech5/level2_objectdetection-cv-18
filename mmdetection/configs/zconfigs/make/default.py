_base_ = [
    "../_base_/models/faster_rcnn_r50_fpn.py",
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/custom_schedule.py",
    "../_base_/custom_runtime.py",
]

# ---------------------------------------- MODEL ----------------------------------------

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=10,
        )
    )
)

# ---------------------------------------- PIPELINE ----------------------------------------

"""
Reference: https://github1s.com/wooseok-shin/Lesion-Detection-3rd-place-solution/blob/main/mmdet_configs/exp8_centernet_resnet18_dcnv2_140e_coco_tta.py#L44-L51
"""

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_train_transforms = [
    dict(type="ShiftScaleRotate", shift_limit=0.0625, scale_limit=0.15, rotate_limit=15, p=0.4),
    dict(type="RandomBrightnessContrast", brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    dict(type="IAAAffine", shear=(-10.0, 10.0), p=0.4),
    dict(type="Blur", p=1.0, blur_limit=7),
    dict(type="CLAHE", p=0.5),
    dict(type="Equalize", mode="cv", p=0.4),
    dict(
        type="OneOf",
        transforms=[
            dict(type="GaussianBlur", p=1.0, blur_limit=7),
            dict(type="MedianBlur", p=1.0, blur_limit=7),
        ],
        p=0.4,
    ),
    dict(type="MixUp", p=0.2, lambd=0.5),
    dict(type="RandomRotate90", p=0.5),
    dict(type="CLAHE", p=0.5),
    dict(type="InvertImg", p=0.5),
    dict(type="Equalize", mode="cv", p=0.4),
    dict(type="MedianBlur", blur_limit=3, p=0.1),
]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    # Albumentations.
    # dict(
    #     type="Albu",
    #     transforms=albu_train_transforms,
    #     bbox_params=dict(
    #         type="BboxParams", format="pascal_voc", label_fields=["gt_labels"], min_visibility=0.0, filter_lost_elements=True
    #     ),
    #     keymap=dict(img="image", gt_bboxes="bboxes"),
    #     update_pad_shape=False,
    #     skip_img_without_anno=True,
    # ),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# ---------------------------------------- DATASET ----------------------------------------

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        pipeline=train_pipeline,
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# ---------------------------------------- OPTIMIZER/SCHEDULER ----------------------------------------

"""
Reference: https://mmdetection.readthedocs.io/en/v2.11.0/tutorials/customize_runtime.html
Example: optimizer = dict(type='Adam', lr=0.0003, weight_decay=0.0001)
"""

optimizer = dict(type="SGD", lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

"""
Reference: https://mmdetection.readthedocs.io/en/v2.11.0/tutorials/customize_runtime.html
Example:
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
"""

lr_config = dict(
    policy="step", warmup="linear", warmup_iters=1000, warmup_ratio=1.0 / 1000, step=[18, 24]
)  # the real step is [18*5, 24*5]
runner = dict(max_epochs=5)  # the real epoch is 28*5=140

# ---------------------------------------- RUNTIME ----------------------------------------

"""
Example: load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
"""
load_from = None
