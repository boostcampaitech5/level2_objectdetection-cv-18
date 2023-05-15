import argparse
import json
import os
import shutil
import subprocess

import yaml
from ultralytics import YOLO

ROOT_PATH = "/opt/ml/dataset"
IMAGE_PATH = os.path.join(ROOT_PATH, "images")
LABEL_PATH = os.path.join(ROOT_PATH, "labels")
CONFIG_PATH = "/opt/ml/baseline/ultralytics/ultralytics/yolo/cfg/zconfigs"

if __name__ == "__main__":
    # argparse 모듈을 사용하여 스크립트 실행 시 사용자로부터 입력받은 인자들을 파싱
    parser = argparse.ArgumentParser(description="YOLO v8 학습 코드")

    parser.add_argument("--yaml_file", help="학습 파라미터를 저장하는 yaml 파일의 이름", required=True)
    args = parser.parse_args()
    with open(os.path.join(CONFIG_PATH, args.yaml_file), "r") as config_yaml_file:
        config = yaml.load(config_yaml_file, Loader=yaml.FullLoader)

    parser.add_argument("--train_json_path", default=config["train_json_path"], help="학습 json파일 경로")
    parser.add_argument("--valid_json_path", default=config["valid_json_path"], help="검증 json파일 경로")
    args = parser.parse_args()

    del config["train_json_path"], config["valid_json_path"]

    # ---------------------------------------- DATA SPLIT ----------------------------------------

    print("-" * 10 + "| 데이터 split |" + "-" * 10)

    # 학습 데이터와 검증 데이터를 json 파일로부터 읽어와서 이미지 파일 경로를 추출
    with open(args.train_json_path, "r") as train_json_file:
        train_data = json.load(train_json_file)
    with open(args.valid_json_path, "r") as valid_json_file:
        valid_data = json.load(valid_json_file)

    train_image_paths = [image["file_name"] for image in train_data["images"]]
    valid_image_paths = [image["file_name"] for image in valid_data["images"]]

    len_train_data = len(train_image_paths)
    len_valid_data = len(valid_image_paths)

    # 분할된 데이터 수 출력
    print("split된 파일 갯수")
    print(f"train: {len_train_data}")
    print(f"valid: {len_valid_data}\n")

    os.makedirs(IMAGE_PATH, exist_ok=True)
    os.makedirs(os.path.join(IMAGE_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(IMAGE_PATH, "valid"), exist_ok=True)

    os.makedirs(LABEL_PATH, exist_ok=True)
    os.makedirs(os.path.join(LABEL_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(LABEL_PATH, "valid"), exist_ok=True)

    # train 폴더와 valid 폴더로 분리하여 복사
    for train_image_path in train_image_paths:
        origin_path = os.path.join(ROOT_PATH, train_image_path)
        moved_path = os.path.join(IMAGE_PATH, "train")
        shutil.copy2(origin_path, moved_path)
    for valid_image_path in valid_image_paths:
        origin_path = os.path.join(ROOT_PATH, valid_image_path)
        moved_path = os.path.join(IMAGE_PATH, "valid")
        shutil.copy2(origin_path, moved_path)

    len_train_files = len(os.listdir(os.path.join(IMAGE_PATH, "train")))
    len_valid_files = len(os.listdir(os.path.join(IMAGE_PATH, "valid")))

    # train/valid 폴더 내 이미지 파일 수 출력
    print("폴더 내 파일 갯수")
    print(f"train: {len_train_files}")
    print(f"valid: {len_valid_files}\n")

    # 분할된 데이터와 train/valid 폴더 내 이미지 파일 수가 일치하는지 검사
    assert len_train_data == len_train_files and len_valid_data == len_valid_files, "데이터 유효성 검사 실패"
    print("데이터 유효성 검사 성공\n")
    print("split된 데이터를 기반으로 COCO 포맷에서 YOLO 포맷으로 label을 변환합니다")
    print("coco2yolo 실행\n")

    train_command = [
        "conda",
        "run",
        "-n",
        "detection",
        "python3",
        "/opt/ml/baseline/ultralytics/coco2yolo.py",
        "-j",
        args.train_json_path,
        "-o",
        os.path.join(LABEL_PATH, "train"),
    ]
    valid_command = [
        "conda",
        "run",
        "-n",
        "detection",
        "python3",
        "/opt/ml/baseline/ultralytics/coco2yolo.py",
        "-j",
        args.valid_json_path,
        "-o",
        os.path.join(LABEL_PATH, "valid"),
    ]
    subprocess.run(train_command, check=True)
    subprocess.run(valid_command, check=True)

    # ---------------------------------------- MODEL ----------------------------------------

    os.chdir("/opt/ml/baseline/ultralytics")
    model = YOLO(config["model"])

    del config["model"]
    config["name"] = args.yaml_file[:-5]

    model.train(**config)
