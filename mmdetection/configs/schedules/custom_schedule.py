# optimizer
optimizer = dict(type='Adam', lr=0.001, weight_decay=0.0001)  # 어떤 optimizer를 사용하는 지
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',  # 어떤 스케줄러를 사용하는지, 스케줄러 마다 인자가 다를 것
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])  # int인 경우 decay하는 step의 간격, list인 경우 decay하는 시점

runner = dict(type='EpochBasedRunner', max_epochs=12)
