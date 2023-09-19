#!/bin/bash

export PYTHONPATH=$(pwd)

PERSON_CFG=configs/coco2014_reduced/coco_person/yolox_s_8x8_300e_coco.py
PERSON_CKPT=/storage/project_data/robust_object_detection_paper/work_dirs/coco_person/yolox/best_bbox_mAP_epoch_290.pth

PARTS_CFG=configs/coco2014_reduced/coco_parts/yolox_s_8x8_300e_coco.py
PARTS_CKPT=/storage/project_data/robust_object_detection_paper/work_dirs/coco_parts/yolox/best_bbox_mAP_epoch_295.pth

# coco dataset tradeoff
SCORE_THRESH_PERSON=0.4900469481945038
SCORE_THRESH_PARTS="0.564262330532074 0.5373225808143616 0.2505517303943634 0.31033051013946533 0.4662320911884308 0.4351080656051636 0.4802977442741394 0.5778014063835144"

IMAGE_PATH=path_to_image
OUT_FILE_PATH=detections.jpg

python3 src/demo.py --image-path ${IMAGE_PATH} --config-person ${PERSON_CFG} --checkpoint-person ${PERSON_CKPT} --config-parts ${PARTS_CFG} --checkpoint-parts ${PARTS_CKPT} --score-thresh-person ${SCORE_THRESH_PERSON} --score-thresh-parts ${SCORE_THRESH_PARTS} --out ${OUT_FILE_PATH}