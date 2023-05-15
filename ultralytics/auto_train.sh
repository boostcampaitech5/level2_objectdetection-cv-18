#!/bin/bash
source activate detection # 가상환경 activate
CONFIG_PATH="/opt/ml/baseline/ultralytics/ultralytics/yolo/cfg/zconfigs"
CONFIG_ENDS_PATH="/opt/ml/baseline/ultralytics/ultralytics/yolo/cfg/zconfigs/ends"

while true; do
    configs=($(ls ${CONFIG_PATH} | grep ".yaml$")) # ls로 config 폴더 안의 모든 파일의 목록을 얻고 grep으로 .yaml이 들어간 파일만 추출
    if [ ${#configs[@]} -gt 0 ]; then # 그렇게 얻어진 config의 길이가 0보다 크다면 (greater)
        echo "Processing config: ${configs[0]}" # echo로 실행될 config명을 출력
        python3 train.py --yaml_file ${configs[0]} # 파이썬 실행 코드
        mv ${CONFIG_PATH}/${configs[0]} ${CONFIG_ENDS_PATH}/${configs[0]} # 실행이 끝난 config는 config에서 config/ends로 이동
    else # config의 길이가 0보다 작거나 같은 경우
        sleep 5 # 5초를 쉰다
    fi
done
