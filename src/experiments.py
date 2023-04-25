import argparse
import os, json
import torch

from pycocotools.coco import COCO
from typing import List
from pathlib import Path
import numpy as np
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt

from src.tools.utils import map_person, map_multilabel, map_reduced_DensePose, bbox_intersection, bbox_area
from src.tools.evaluation_protocol import decision_strategy_1, decision_strategy_2, compute_TP_FP_FN_detections_per_image

# Configurations for experiments
OVERLAP_THRESHOLDS = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
IOU=0.5

MAPPING_PARTS = None
MAPPING_PERSON = map_person       

def per_image_experiment(args, im_ids: List[int], dataset: COCO, Dt: COCO, Dt_parts: COCO, iou_threshold: float, overlap_threshold: float):
    """Computes the precision, recall, and MCC for the per-image experiment"""

    l_FP_detected, l_FN_detected = [], []
    l_FP_present, l_FN_present = [], []

    for id in im_ids:

        TPS, FPS, FNS = compute_TP_FP_FN_detections_per_image(dataset=dataset, im_id=id, Dt=Dt, score_threshold=args.score_thresh_person, iou_threshold=iou_threshold)

        nFPs = len(FPS)
        nFNs = len(FNS)

        FP_DETECTED, FN_DETECTED = decision_strategy_1(dataset=dataset, im_id=id, Dt=Dt, Dt_parts=Dt_parts, 
                score_thresh=args.score_thresh_person, score_thresh_parts=args.score_thresh_parts, overlap_threshold=overlap_threshold)
            
        l_FP_detected.append(FP_DETECTED)
        l_FN_detected.append(FN_DETECTED)
        l_FP_present.append(nFPs > 0)
        l_FN_present.append(nFNs > 0)
            
    l_FP_detected = np.array(l_FP_detected)
    l_FN_detected = np.array(l_FN_detected)
    l_FP_present = np.array(l_FP_present)
    l_FN_present = np.array(l_FN_present)
    
    tp = np.logical_and(l_FP_detected, l_FP_present).sum()
    fp = np.logical_and(l_FP_detected, np.logical_not(l_FP_present)).sum()
    fn = np.logical_and(np.logical_not(l_FP_detected), l_FP_present).sum()
    tn = np.logical_and(np.logical_not(l_FP_detected), np.logical_not(l_FP_present)).sum()
    
    precision_FP = tp/(tp+fp+1e-8)
    recall_FP = tp/(tp+fn+1e-8)
    
    MCC_FP = matthews_corrcoef(y_true=l_FP_present, y_pred=l_FP_detected)
    MCC_FN = matthews_corrcoef(y_true=l_FN_present, y_pred=l_FN_detected)
    
    tp_FP = tp
    fp_FP = fp
    errors_FP = tp+fn
        
    tp = np.logical_and(l_FN_detected, l_FN_present).sum()
    fp = np.logical_and(l_FN_detected, np.logical_not(l_FN_present)).sum()
    fn = np.logical_and(np.logical_not(l_FN_detected), l_FN_present).sum()
    tn = np.logical_and(np.logical_not(l_FN_detected), np.logical_not(l_FN_present)).sum()
    
    tp_FN = tp
    fp_FN = fp
    errors_FN = tp+fn
    
    precision_FN = tp/(tp+fp+1e-8)
    recall_FN = tp/(tp+fn+1e-8)
        
    return {"MCC_FP": MCC_FP, "MCC_FN": MCC_FN, "precision_FP": precision_FP, "precision_FN": precision_FN, "recall_FP": recall_FP, "recall_FN": recall_FN, 
            "tp_FP": tp_FP, "tp_FN": tp_FN, "fp_FP": fp_FP, "fp_FN": fp_FN, "errors_FP": errors_FP, "errors_FN": errors_FN, "nImages": len(im_ids)}
                  
def per_object_experiment(args, im_ids: List[int], dataset: COCO, Dt: COCO, Dt_parts: COCO, iou_threshold, overlap_threshold):
    """Computes the balances for the per-object experiment"""
    
    n_detected_TPs = 0
    n_detected_FPs = 0
    n_detected_FNs = 0
    n_missed_TPs = 0
    n_ghost_parts = 0
    
    n_TPs = 0
    n_FPs = 0
    n_FNs = 0

    for id in im_ids:
        
        # we need all gt boxes for the ghost parts
        img = dataset.loadImgs(id)[0]

        ann_ids = dataset.getAnnIds(imgIds=img['id'])
        annotations = dataset.loadAnns(ann_ids)

        GT_boxes = np.array([ann['bbox'] for ann in annotations]) # also consider iscrowd annotations
        GT_labels = np.array([ann['category_id'] for ann in annotations]) # also consider iscrowd annotations
        
        # only consider person annotations
        keep = (GT_labels == 1)
        GT_labels = GT_labels[keep]
        GT_boxes = GT_boxes[keep].tolist()
        
        TPS, FPS, FNS = compute_TP_FP_FN_detections_per_image(dataset=dataset, im_id=id, Dt=Dt, score_threshold=args.score_thresh_person, iou_threshold=iou_threshold)

        l_TPs, l_FPs, l_FNs = decision_strategy_2(dataset=dataset, im_id=id, Dt=Dt, Dt_parts=Dt_parts, 
                score_thresh=args.score_thresh_person, score_thresh_parts=args.score_thresh_parts, overlap_threshold=overlap_threshold)
        
        n_TPs += len(TPS)
        n_FPs += len(FPS)
        n_FNs += len(FNS)
        
        # check how many TP detections got correctly classified as TP by the monitor
        for person_box in TPS:
            if person_box[:4] in l_TPs:
                n_detected_TPs += 1
            else:
                n_missed_TPs += 1
        
        # check how many FP detections got correctly classified as FP by the monitor
        for person_box in FPS:
            if person_box[:4] in l_FPs:
                n_detected_FPs += 1
                
        # check whether a missed person got detected by the monitor
        for person_box in FNS:
            for part_box in l_FNs:
                overlap =  bbox_intersection(np.array(part_box), np.array(person_box[:4]), x1y1x2y2=False)
                area_part = bbox_area(np.array(part_box), x1y1x2y2=False)
                area_person = bbox_area(np.array(person_box[:4]), x1y1x2y2=False)
                area = min(area_part, area_person)
                if overlap >= (1.-overlap_threshold)*area:
                    n_detected_FNs += 1
                    break
        
        all_boxes = []
        all_boxes.extend(GT_boxes)
        all_boxes.extend(TPS)
        all_boxes.extend(FPS)
        
        # check how many ghost body-parts are produced by the monitor
        for part_box in l_FNs:
            
            is_GP = True
            for person_box in all_boxes:
                
                overlap =  bbox_intersection(np.array(part_box), np.array(person_box[:4]), x1y1x2y2=False)
                area_part = bbox_area(np.array(part_box), x1y1x2y2=False)
                area_person = bbox_area(np.array(person_box[:4]), x1y1x2y2=False)
                area = min(area_part, area_person)
                
                if overlap >= (1.-overlap_threshold)*area:
                    is_GP = False
                    break
            
            if is_GP:
                n_ghost_parts += 1
                
    print(f'Detected FPs: ({n_detected_FPs}/{n_FPs} --> {np.round(n_detected_FPs/n_FPs*100, decimals=1)}%), missed TPs: ({n_missed_TPs}/{n_TPs}--> {np.round(n_missed_TPs/n_TPs*100, decimals=1)}%), Balance: {n_detected_FPs-n_missed_TPs}')
    print(f'Detected FNs: ({n_detected_FNs}/{n_FNs} --> {np.round(n_detected_FNs/n_FNs*100, decimals=1)}%), ghost parts: {n_ghost_parts}), Balance: {n_detected_FNs-n_ghost_parts}')
    return {"tps": int(n_TPs),"fps": int(n_FPs),"fns": int(n_FNs), "detected_fps": int(n_detected_FPs), "detected_fns": int(n_detected_FNs), "missed_tps": int(n_missed_TPs), "ghost_parts": n_ghost_parts}

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Experiments')

    args.add_argument('--experiment', type=str,
                      help='Experiments, options: per_image, per_object')
    args.add_argument('--path_imgs', type=str)
    args.add_argument('--path_anns', type=str)
    args.add_argument('--path_dets_person', type=str)
    args.add_argument('--path_dets_parts', type=str)
    args.add_argument('--save_dir', type=str)
    args.add_argument('--score_thresh_person', type=float)
    args.add_argument('--score_thresh_parts', nargs="+")
    args = args.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.score_thresh_parts = [float(score_thresh) for score_thresh in args.score_thresh_parts]
    
    if len(args.score_thresh_parts) == 1:
        MAPPING_PARTS = map_person
    elif len(args.score_thresh_parts) == 8:
        MAPPING_PARTS = map_reduced_DensePose
    elif len(args.score_thresh_parts) == 9:
        MAPPING_PARTS = map_multilabel
    else:
        raise NotImplementedError
    
    dataset = COCO(args.path_anns)
    im_ids = dataset.getImgIds()

    Dt = dataset.loadRes(args.path_dets_person)
    Dt_parts = dataset.loadRes(args.path_dets_parts)

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        
    if args.experiment == 'per_image': 
        
        l_MCC_FP, l_MCC_FN, l_PREC_FP, l_PREC_FN, l_REC_FP, l_REC_FN = [], [], [], [], [], []
        stats = {"FP": {}, "FN": {}}
        
        for overlap in OVERLAP_THRESHOLDS:
            
            per_image_results = per_image_experiment(args=args, im_ids=im_ids, dataset=dataset, Dt=Dt, Dt_parts=Dt_parts, iou_threshold=IOU, overlap_threshold=overlap)
            
            l_MCC_FP.append(per_image_results["MCC_FP"])
            l_MCC_FN.append(per_image_results["MCC_FN"])
            l_PREC_FP.append(per_image_results["precision_FP"])
            l_PREC_FN.append(per_image_results["precision_FN"])
            l_REC_FP.append(per_image_results["recall_FP"])
            l_REC_FN.append(per_image_results["recall_FN"])
            
            stats["FP"][overlap] = {"tp": int(per_image_results["tp_FP"]), "fp": int(per_image_results["fp_FP"]), "precision": float(per_image_results["precision_FP"]), 
                                    "recall": float(per_image_results["recall_FP"]), "MCC": float(per_image_results["MCC_FP"]), "N": int(per_image_results["errors_FP"]), 
                                    "nImages": per_image_results["nImages"]}
            stats["FN"][overlap] = {"tp": int(per_image_results["tp_FN"]), "fp": int(per_image_results["fp_FN"]), "precision": float(per_image_results["precision_FN"]), 
                                    "recall": float(per_image_results["recall_FN"]), "MCC": float(per_image_results["MCC_FN"]), "N": int(per_image_results["errors_FN"]), 
                                    "nImages": per_image_results["nImages"]}
            
        plt.figure()
        plt.title(f'Stats (False Positives) for IoU={IOU}, max = {OVERLAP_THRESHOLDS[l_MCC_FP.index(max(l_MCC_FP))]}')
        plt.xlabel('Overlap Thresholds')
        plt.plot(OVERLAP_THRESHOLDS, l_MCC_FP, label='MCC')
        plt.plot(OVERLAP_THRESHOLDS, l_PREC_FP, label='Precision')
        plt.plot(OVERLAP_THRESHOLDS, l_REC_FP, label='Recall')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, f'analysis_FP_max_{OVERLAP_THRESHOLDS[l_MCC_FP.index(max(l_MCC_FP))]}.jpg'))
        plt.close()
        
        plt.figure()
        plt.title(f'Stats (False Negatives) for IoU={IOU}, max = {OVERLAP_THRESHOLDS[l_MCC_FN.index(max(l_MCC_FN))]}')
        plt.xlabel('Overlap Thresholds')
        plt.plot(OVERLAP_THRESHOLDS, l_MCC_FN, label='MCC')
        plt.plot(OVERLAP_THRESHOLDS, l_PREC_FN, label='Precision')
        plt.plot(OVERLAP_THRESHOLDS, l_REC_FN, label='Recall')
        plt.legend()
        plt.savefig(os.path.join(args.save_dir, f'analysis_FN_max_{OVERLAP_THRESHOLDS[l_MCC_FN.index(max(l_MCC_FN))]}.jpg'))
        plt.close()

        # save values in .json file
        with open(os.path.join(args.save_dir, f'per_image_experiment.json'), 'w') as f:
            json.dump(stats, f, indent=6)
            
    elif args.experiment == 'per_object': 
        
        balance_FPs, balance_FNs = [], []
        balances = {}
        
        for overlap_thresh in OVERLAP_THRESHOLDS:
            
            print("---------------------------------------")
            print(f'Overlap threshold = {overlap_thresh}')
            print("---------------------------------------")
            
            stats = per_object_experiment(args=args, im_ids=im_ids, dataset=dataset, Dt=Dt, Dt_parts=Dt_parts, iou_threshold=IOU, overlap_threshold=overlap_thresh)
            balances[overlap_thresh] = stats
            
            balance_FPs.append(stats["detected_fps"]-stats["missed_tps"])
            balance_FNs.append(stats["detected_fns"]-stats["ghost_parts"])
            
        plt.figure()
        plt.title(f'Balance (det. FPs - missed TPs) for IoU={IOU}, max = {OVERLAP_THRESHOLDS[balance_FPs.index(max(balance_FPs))]}')
        plt.xlabel('Overlap Thresholds')
        plt.plot(OVERLAP_THRESHOLDS, balance_FPs)
        plt.savefig(os.path.join(args.save_dir, f'balance_FP_max_{OVERLAP_THRESHOLDS[balance_FPs.index(max(balance_FPs))]}.jpg'))
        plt.close()
        
        plt.figure()
        plt.title(f'Balance (det. FNs - ghost parts) for IoU={IOU}, max = {OVERLAP_THRESHOLDS[balance_FNs.index(max(balance_FNs))]}')
        plt.xlabel('Overlap Thresholds')
        plt.plot(OVERLAP_THRESHOLDS, balance_FNs)
        plt.savefig(os.path.join(args.save_dir, f'balance_FN_max_{OVERLAP_THRESHOLDS[balance_FNs.index(max(balance_FNs))]}.jpg'))
        plt.close()

        
        # save values in .json file
        with open(os.path.join(args.save_dir, f'per_object_experiment.json'), 'w') as f:
            json.dump(balances, f, indent=6)
    else:
        raise ValueError('Invalid experiment.')
