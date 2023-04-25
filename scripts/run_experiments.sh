#!/bin/bash

export PYTHONPATH=$(pwd)

###################################################################################################################################
# Configuration
###################################################################################################################################
# Replace with your own path

ROOT_DIR=IV23

DATASETS=(voc2010 coco2014_full)
IMGS_DIRS=(/storage/yavis/datasets/PASCAL_VOC_2010/VOCdevkit/VOC2010/JPEGImages /storage/datasets/clean/coco_2014/images/val2014) 
ANNS_FILE=("trainval.json" "val.json")
MODELS=(fcos yolox cascade_rcnn)

SAVE_DIR=${ROOT_DIR}/results/
DETS_DIR=${ROOT_DIR}/detections/
ANNS_DIR=${ROOT_DIR}/annotations/

###################################################################################################################################
# Baseline
###################################################################################################################################
# confidence thresholds have been chosen such that best precision-recall trade-off is achieved

SCORE_THRESHS_PERSON=(0.31164032220840454 0.4900469481945038 0.7155476808547974)

cascade_rcnn_theshs="0.790003776550293"
fcos_theshs="0.3388860821723938"
yolox_theshs="0.3726857006549835"

SCORE_THRESHS_PARTS=("$fcos_theshs" "$yolox_theshs"  "$cascade_rcnn_theshs")

d_counter=0
for dataset in ${DATASETS[@]}
do
    m_counter=0
    for model in ${MODELS[@]}
    do
        echo "Running experiments for $model on the dataset $dataset ..."

        ANNS_DATASET=${ANNS_DIR}$dataset
        DETS_DATASET=${DETS_DIR}$dataset
        SAVE_DIR_DATASET=${SAVE_DIR}$dataset/baseline/tradeoff/$model

        python src/prepare_detections.py --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dir_person ${DETS_DATASET}/coco_person --path_dir_baseline ${DETS_DATASET}/coco_baseline --json_file $model.bbox.json

        python src/experiments.py --experiment per_image --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_baseline/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_baseline/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        python src/experiments.py --experiment per_object --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_baseline/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_baseline/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        
        m_counter=$((m_counter+1))
    done
    d_counter=$((d_counter+1))
done

###################################################################################################################################
# MultiDet
###################################################################################################################################
# confidence thresholds have been chosen such that best precision-recall trade-off is achieved

SCORE_THRESHS_PERSON=(0.31164032220840454 0.4900469481945038 0.7155476808547974)
cascade_rcnn_theshs="0.6903649568557739 0.6256222128868103 0.32768628001213074 0.37532055377960205 0.5463505387306213 0.5984303951263428 0.6151552200317383 0.8304176330566406"
fcos_theshs="0.26400694251060486 0.2955682575702667 0.23296034336090088 0.2704211473464966 0.25903254747390747 0.3041515350341797 0.30829891562461853 0.3475559651851654"
yolox_theshs="0.6195120215415955 0.5393099784851074 0.29129311442375183 0.33345621824264526 0.5621513724327087 0.5291999578475952 0.5523180961608887 0.551236629486084"
SCORE_THRESHS_PARTS=("$fcos_theshs" "$yolox_theshs" "$cascade_rcnn_theshs")

d_counter=0
for dataset in ${DATASETS[@]}
do
    m_counter=0
    for model in ${MODELS[@]}
    do
        echo "Running experiments for $model on the dataset $dataset ..."
        echo "${IMGS_DIRS[$d_counter]}"

        ANNS_DATASET=${ANNS_DIR}$dataset
        DETS_DATASET=${DETS_DIR}$dataset
        SAVE_DIR_DATASET=${SAVE_DIR}$dataset/two_detectors/tradeoff/$model

        python src/experiments.py --experiment per_image --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        python src/experiments.py --experiment per_object --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        
        m_counter=$((m_counter+1))
    done
    d_counter=$((d_counter+1))
done

###################################################################################################################################
# SingleDet
###################################################################################################################################
# confidence thresholds have been chosen such that best precision-recall trade-off is achieved

SCORE_THRESHS_PERSON=(0.2731368839740753 0.664621114730835 0.7709288001060486)

cascade_rcnn_theshs="-1 0.6557528972625732 0.5206194519996643 0.32372528314590454 0.4205414056777954 0.5990193486213684 0.5922345519065857 0.5629574656486511 0.6408008337020874"
fcos_theshs="-1 0.275457501411438 0.2911315858364105 0.2192033976316452 0.26748234033584595 0.26560455560684204 0.2721535861492157 0.2832152843475342 0.2640025317668915"
yolox_theshs="-1 0.5154919624328613 0.5214234590530396 0.2727903425693512 0.3203616142272949 0.45618101954460144 0.4982021749019623 0.5038536190986633 0.6092017292976379"

SCORE_THRESHS_PARTS=("$fcos_theshs" "$yolox_theshs" "$cascade_rcnn_theshs")

d_counter=0
for dataset in ${DATASETS[@]}
do
    m_counter=0
    for model in ${MODELS[@]}
    do
        echo "Running experiments for $model on the dataset $dataset ..."

        ANNS_DATASET=${ANNS_DIR}$dataset
        DETS_DATASET=${DETS_DIR}$dataset
        SAVE_DIR_DATASET=${SAVE_DIR}$dataset/one_detector/tradeoff/$model

        python src/prepare_detections.py --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dir_multilabel ${DETS_DATASET}/coco_multilabel --json_file $model.bbox.json

        python src/experiments.py --experiment per_image --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_multilabel/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_multilabel/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        python src/experiments.py --experiment per_object --path_imgs ${IMGS_DIRS[$d_counter]}  --path_anns ${ANNS_DATASET}/coco_person/${ANNS_FILE[$d_counter]} --path_dets_person ${DETS_DATASET}/coco_multilabel/coco_person/$model.bbox.json --path_dets_parts ${DETS_DATASET}/coco_multilabel/coco_parts/$model.bbox.json --save_dir ${SAVE_DIR_DATASET} --score_thresh_person ${SCORE_THRESHS_PERSON[$m_counter]} --score_thresh_parts ${SCORE_THRESHS_PARTS[$m_counter]}
        
        m_counter=$((m_counter+1))
    done
    d_counter=$((d_counter+1))
done

# Summarize results
python src/latex.py --root ${SAVE_DIR}
