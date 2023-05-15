# 모듈 import
import argparse

import wandb
from mmcv import Config
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset  # , build_dataloader, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.utils import get_device

wandb.login()

parser = argparse.ArgumentParser(description="모델 학습")
parser.add_argument("--config_path", default="expbase_faster_rcnn_r50_fpn", help="config 파일 경로")
parser.add_argument("--train_data_name", default="train_.json", help="train data 파일 경로")
parser.add_argument("--validation_data_name", default="val_.json", help="validation data 파일 경로")
args = parser.parse_args()

classes = (
    "General trash",
    "Paper",
    "Paper pack",
    "Metal",
    "Glass",
    "Plastic",
    "Styrofoam",
    "Plastic bag",
    "Battery",
    "Clothing",
)

# TODO
config_path = f"/opt/ml/baseline/mmdetection/configs/zconfigs/{args.config_path}.py"  # config 경로 설정
root = "/opt/ml/dataset/"  # 경로 설정
work_dir = f"/opt/ml/baseline/mmdetection/work_dirs/{args.config_path}"  # workdir 경로 설정

# config file 들고오기
cfg = Config.fromfile(config_path)


train_file = f"{args.train_data_name}"
val_file = f"{args.validation_data_name}"
# test_file = "test.json"


# dataset config 수정

# train_resize = (512, 512)
# test_resize = (512, 512)

cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + train_file  # train json 정보
# cfg.data.train.pipeline[2]["img_scale"] = train_resize  # Resize (삭제)  이거 지워야해

cfg.data.val.classes = classes
cfg.data.val.img_prefix = root
cfg.data.val.ann_file = root + val_file  # val json 정보

# cfg.data.test.classes = classes
# cfg.data.test.img_prefix = root
# cfg.data.test.ann_file = root + test_file  # test json 정보
# cfg.data.test.pipeline[1]["img_scale"] = test_resize  # Resize (삭제)

# Runtime 설정

# batch_size = 32
# seed = 18
# epochs = 50

# cfg.data.samples_per_gpu = batch_size
# cfg.runner.max_epochs = epochs
cfg.seed = 18

cfg.gpu_ids = [0]

cfg.work_dir = work_dir

# model 설정

# cfg.model.roi_head.bbox_head.num_classes = len(classes)

# cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)  # 너무 큰 grad를 clip해 준다.
cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)  # 모델의 checkpoint를 몇 개 저장할지
cfg.device = get_device()

# model 평가 설정
cfg.evaluation.save_best = "bbox_mAP_50"  # bestmodel 저장 기준, bbox_mAP도 있다.
cfg.evaluation.metric = "bbox"
cfg.evaluation.classwise = True  # 각 class 별 AP 계산

# wandb config 설정
cfg.log_config.hooks[1].init_kwargs.config = cfg
cfg.log_config.hooks[1].init_kwargs.name = f"{args.config_path}"

# build_dataset
datasets = [build_dataset(cfg.data.train)]

# dataset 확인, train, val 비율 맞는지 확인하기

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=True)
