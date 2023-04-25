#!/bin/bash

export PYTHONPATH=$(pwd)

###################################################################################################################################
# Configuration
###################################################################################################################################
# Replace with your own path

ROOT_DIR=IV23
SAVE_ANNS_DIR=${ROOT_DIR}/annotations
OLD_ANNS_DENSEPOSE=/storage/datasets/clean/DensePose_COCO
OLD_ANNS_COCO=/storage/datasets/clean/coco_2014/annotations
OLD_ANNS_VOC=IV23/annotations
OLD_ANNS_VOC_NAME=voc2010_trainval_cocoformat.json

#################################################### DensePose annotations

DATASET=coco2014_reduced

echo "------------------------------------------------------------"
echo "coco2014_reduced (train)"
echo "------------------------------------------------------------"

python src/generate_annotations.py --anns_mode person --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_train.json --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_person --new_anns_file train.json
python src/generate_annotations.py --anns_mode parts --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_train.json  --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_parts --new_anns_file train.json
python src/generate_annotations.py --anns_mode multilabel --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_train.json --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_multilabel --new_anns_file train.json

echo "------------------------------------------------------------"
echo "coco2014_reduced (valminusminival)"
echo "------------------------------------------------------------"

python src/generate_annotations.py --anns_mode person --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_valminusminival.json  --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_person --new_anns_file valminusminival.json
python src/generate_annotations.py --anns_mode parts --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_valminusminival.json  --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_parts --new_anns_file valminusminival.json
python src/generate_annotations.py --anns_mode multilabel --old_anns ${OLD_ANNS_DENSEPOSE}/densepose_coco_2014_valminusminival.json  --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_multilabel --new_anns_file valminusminival.json

#################################################### Full coco2014 validation annotations

DATASET=coco2014_full

echo "------------------------------------------------------------"
echo "coco2014_full (val2014)"
echo "------------------------------------------------------------"

python src/generate_annotations.py --anns_mode all_persons_person --old_anns ${OLD_ANNS_COCO}/instances_val2014.json --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_person --new_anns_file val.json
python src/generate_annotations.py --anns_mode all_persons_parts --old_anns ${OLD_ANNS_COCO}/instances_val2014.json --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_parts --new_anns_file val.json
python src/generate_annotations.py --anns_mode all_persons_multilabel --old_anns ${OLD_ANNS_COCO}/instances_val2014.json --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_multilabel --new_anns_file val.json

#################################################### PascalVOC2010 annotatsions

DATASET=voc2010

echo "------------------------------------------------------------"
echo "voc2010 (trainval)"
echo "------------------------------------------------------------"

python src/generate_annotations.py --anns_mode voc2010_person --old_anns ${OLD_ANNS_VOC}/${OLD_ANNS_VOC_NAME} --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_person --new_anns_file trainval.json
python src/generate_annotations.py --anns_mode voc2010_parts --old_anns ${OLD_ANNS_VOC}/${OLD_ANNS_VOC_NAME} --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_parts --new_anns_file trainval.json
python src/generate_annotations.py --anns_mode voc2010_multilabel --old_anns ${OLD_ANNS_VOC}/${OLD_ANNS_VOC_NAME} --new_anns_dir ${SAVE_ANNS_DIR}/${DATASET}/coco_multilabel --new_anns_file trainval.json