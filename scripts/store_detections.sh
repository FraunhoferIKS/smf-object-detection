#!/bin/bash

export PYTHONPATH=$(pwd)

###################################################################################################################################
# Configuration
###################################################################################################################################
# Replace with your own path

CONFIG_DIR=configs
ROOT_DIR=IV23
DETS_SAVE_DIR=${ROOT_DIR}/detections
WORK_DIR=/storage/project_data/robust_object_detection_paper/work_dirs
SCORE_THRESH=0.05

DATASETS=(coco2014_reduced coco2014_full voc2010)
MODEL_C1=(fcos yolox cascade_rcnn)
MODEL_C2=(fcos_r50_caffe_fpn_gn-head_1x_coco.py yolox_s_8x8_300e_coco.py cascade_rcnn_r50_fpn.py)

###################################################################################################################################

for DATASET in ${DATASETS}
do
    counter=0
    for i in ${MODEL_C1[@]}
    do
        echo "Storing results for $i on the dataset ${DATASET} ..."

        MODE=coco_person
        WEIGHTS=-1
        WEIGHTS_DIR=${WORK_DIR}/${MODE}/$i
        for filename in ${WEIGHTS_DIR}/*.pth 
        do
            echo ${filename}
            if [[ $filename == *"best_bbox_mAP"* ]]
            then
                WEIGHTS=${filename}
                break
            fi
        done
        python mmdetection/tools/test.py ${CONFIG_DIR}/${DATASET}/${MODE}/${MODEL_C2[$counter]} ${WEIGHTS} --format-only --eval-options jsonfile_prefix=${DETS_SAVE_DIR}/${DATASET}/${MODE}/$i --show-score-thr ${SCORE_THRESH}

        MODE=coco_multilabel
        WEIGHTS=-1
        WEIGHTS_DIR=${WORK_DIR}/${MODE}/$i
        for filename in ${WEIGHTS_DIR}/*.pth
        do
            echo ${filename}
            if [[ $filename == *"best_bbox_mAP"* ]]
            then
                WEIGHTS=${filename}
                break
            fi
        done
        python mmdetection/tools/test.py ${CONFIG_DIR}/${DATASET}/${MODE}/${MODEL_C2[$counter]} ${WEIGHTS}  --format-only --eval-options jsonfile_prefix=${DETS_SAVE_DIR}/${DATASET}/${MODE}/$i --show-score-thr ${SCORE_THRESH}
    
        MODE=coco_baseline
        WEIGHTS=-1
        WEIGHTS_DIR=${WORK_DIR}/${MODE}/$i
        for filename in ${WEIGHTS_DIR}/*.pth 
        do
            echo ${filename}
            if [[ $filename == *"best_bbox_mAP"* ]]
            then
                WEIGHTS=${filename}
                break
            fi
        done
        python mmdetection/tools/test.py ${CONFIG_DIR}/$DATASET/${MODE}/${MODEL_C2[$counter]} ${WEIGHTS} --format-only --eval-options jsonfile_prefix=${DETS_SAVE_DIR}/$DATASET/${MODE}/$i --show-score-thr ${SCORE_THRESH}
        
        MODE=coco_parts
        WEIGHTS=-1
        WEIGHTS_DIR=${WORK_DIR}/${MODE}/$i
        for filename in ${WEIGHTS_DIR}/*.pth
        do
            echo ${filename}
            if [[ $filename == *"best_bbox_mAP"* ]]
            then
                WEIGHTS=${filename}
                break
            fi
        done
        python mmdetection/tools/test.py ${CONFIG_DIR}/${DATASET}/${MODE}/${MODEL_C2[$counter]} ${WEIGHTS}  --format-only --eval-options jsonfile_prefix=${DETS_SAVE_DIR}/${DATASET}/${MODE}/$i --show-score-thr ${SCORE_THRESH}
    
    counter=$((counter+1))
    done
done