#!/bin/bash

export PYTHONPATH=$(pwd)

###################################################################################################################################
# Configuration
###################################################################################################################################
# Replace with your own path

ROOT_DIR=IV23

SAVE_DIR=${ROOT_DIR}/performance_analysis
DETS_DIR=${ROOT_DIR}/detections/
ANNS_DIR=${ROOT_DIR}/annotations

###################################################################################################################################

MODELS=(fcos yolox cascade_rcnn)

DATASET=coco2014_reduced

for i in ${MODELS[@]}
do
    echo "Running experiments for $i on the dataset ${DATASET} ..."

    # person detector
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/valminusminival.json --path_dets ${DETS_DIR}/${DATASET}/coco_person/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_person/$i --mode person
    # baseline detector
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/valminusminival.json --path_dets ${DETS_DIR}/${DATASET}/coco_baseline/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_baseline/$i --mode person
    # parts detector
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_parts/valminusminival.json --path_dets ${DETS_DIR}/${DATASET}/coco_parts/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_parts/$i --mode parts
    # multilabel detector
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_multilabel/valminusminival.json --path_dets ${DETS_DIR}/${DATASET}/coco_multilabel/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_multilabel/$i --mode multilabel

done

DATASET=coco2014_full
for i in ${MODELS[@]}
do
    echo "Running experiments for $i on the dataset ${DATASET} ..."

    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/val.json --path_dets ${DETS_DIR}/${DATASET}/coco_person/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_person/$i --mode person
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_multilabel/val.json --path_dets ${DETS_DIR}/${DATASET}/coco_multilabel/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_multilabel/$i --mode multilabel
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/val.json --path_dets ${DETS_DIR}/${DATASET}/coco_baseline/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_baseline/$i --mode person
done

DATASET=voc2010
for i in ${MODELS[@]}
do
    echo "Running experiments for $i on the dataset ${DATASET} ..."

    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/trainval.json --path_dets ${DETS_DIR}/${DATASET}/coco_person/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_person/$i --mode person
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_multilabel/trainval.json --path_dets ${DETS_DIR}/${DATASET}/coco_multilabel/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_multilabel/$i --mode multilabel
    python src/analysis.py --path_anns ${ANNS_DIR}/${DATASET}/coco_person/trainval.json --path_dets ${DETS_DIR}/${DATASET}/coco_baseline/$i.bbox.json --save_path ${SAVE_DIR}/${DATASET}/coco_baseline/$i --mode person
done