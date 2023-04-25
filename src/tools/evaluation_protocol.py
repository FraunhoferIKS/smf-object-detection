import numpy as np
from typing import Tuple, List
from pycocotools.coco import COCO
from src.tools.utils import bbox_iou, bbox_intersection, bbox_area

from src.tools.utils import discard_part_detection

def compute_TP_FP_FN_detections_per_image(dataset, im_id: int, Dt: COCO, score_threshold, iou_threshold=0.5) -> Tuple[List, List, List]:
    
    img = dataset.loadImgs(im_id)[0]

    ann_ids = dataset.getAnnIds(imgIds=img['id'])
    annotations = dataset.loadAnns(ann_ids)

    dets_ids = Dt.getAnnIds(imgIds=img['id'])
    detections = Dt.loadAnns(dets_ids)

    boxes = np.array([det['bbox'] for det in detections])
    scores = np.array([det['score'] for det in detections])
    labels = np.array([det['category_id'] for det in detections])

    GT_boxes = np.array([ann['bbox'] for ann in annotations]) # also consider iscrowd annotations
    GT_labels = np.array([ann['category_id'] for ann in annotations]) # also consider iscrowd annotations
    
    # get positive person detections
    keep = scores >= score_threshold
    labels = labels[keep]
    scores = scores[keep]
    boxes = boxes[keep]
    
    # only consider person annotations
    keep = (GT_labels == 1)
    GT_labels = GT_labels[keep]
    GT_boxes = GT_boxes[keep]
    
    if len(GT_labels):
        
        # sort detections
        sorted_ids = np.argsort(scores)
        sorted_ids = sorted_ids[::-1]

        sorted_labels = labels[sorted_ids]
        sorted_scores = scores[sorted_ids]
        sorted_boxes = boxes[sorted_ids]

        FPS, TPS, FNS = [], [], []
        matched_GT_ids = []

        # compute TPs, FPs, FNs
        for i in range(len(sorted_labels)):

            box = sorted_boxes[i, :]
            ious = np.array([bbox_iou(box.tolist(), GT_boxes[j, :].tolist(), x1y1x2y2=False) for j in range(len(GT_labels))])

            # sort ious
            sorted_ids_iou = np.argsort(ious)
            sorted_ids_iou = sorted_ids_iou[::-1]
            sorted_ious = ious[sorted_ids_iou].tolist()
            # sorted_GT_boxes = GT_boxes[sorted_ids_iou]

            matched=False
            for s_i, s_iou in enumerate(sorted_ious):
                if s_iou >= iou_threshold: #  and sorted_ids_iou[s_i] not in matched_GT_ids:
                    TPS.append([box[0], box[1], box[2], box[3], sorted_scores[i], sorted_labels[i]])
                    matched_GT_ids.append(sorted_ids_iou[s_i])
                    matched=True
                    break
                elif s_iou < iou_threshold:
                    break

            if not matched:
                FPS.append([box[0], box[1], box[2], box[3], sorted_scores[i], sorted_labels[i]])

        FNS = [[GT_boxes[j, 0], GT_boxes[j, 1], GT_boxes[j, 2], GT_boxes[j, 3], GT_labels[j]] for j in range(len(GT_labels)) if j not in matched_GT_ids]
               
    else:
        TPS, FNS, FPS = [], [], []
        for i in range(len(labels)):
            FPS.append([boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], scores[i], labels[i]])

    return TPS, FPS, FNS


def compute_TP_FP_FN_detections(dataset, im_ids, Dt_person, score_threshold: float=0.5, iou_threshold: float=0.5, inspect_images: bool=True):
    # Computes TP, FP, FN detections per image and returns Lists including images that contain TP, FP, and FN detections
    
    images2inspect_FNS = []
    images2inspect_FPS = []
    images2inspect_TPS = []

    all_images = []

    for id in im_ids:

        TPS, FPS, FNS = compute_TP_FP_FN_detections_per_image(dataset, im_id=id, Dt=Dt_person, 
                score_threshold=score_threshold, iou_threshold=iou_threshold)

        d_img = {"id": id, "fp": False, "fn": False}

        if len(FNS) > 0:

            d_img["fn"] = True

            tp_boxes = np.array([[tp[0], tp[1], tp[0]+tp[2], tp[1]+tp[3]] for tp in TPS])
            tp_scores = np.array([tp[4] for tp in TPS])
            tp_labels = np.array([tp[5] for tp in TPS])

            fp_boxes = np.array([[fp[0], fp[1], fp[0]+fp[2], fp[1]+fp[3]] for fp in FPS])
            fp_scores = np.array([fp[4] for fp in FPS])
            fp_labels = np.array([fp[5] for fp in FPS])

            fn_boxes = np.array([[fn[0], fn[1], fn[0]+fn[2], fn[1]+fn[3]] for fn in FNS])
            fn_labels = np.array([fn[4] for fn in FNS])

            images2inspect_FNS.append({"id": id, "fn_boxes": fn_boxes, "tp_boxes": tp_boxes, "fp_boxes": fp_boxes})
        
        if len(FPS) > 0:

            d_img["fp"] = True

            tp_boxes = np.array([[tp[0], tp[1], tp[0]+tp[2], tp[1]+tp[3]] for tp in TPS])
            tp_scores = np.array([tp[4] for tp in TPS])

            fp_boxes = np.array([[fp[0], fp[1], fp[0]+fp[2], fp[1]+fp[3]] for fp in FPS])
            fp_scores = np.array([fp[4] for fp in FPS])

            fn_boxes = np.array([[fn[0], fn[1], fn[0]+fn[2], fn[1]+fn[3]] for fn in FNS])

            images2inspect_FPS.append({"id": id, "fn_boxes": fn_boxes, "tp_boxes": tp_boxes, "fp_boxes": fp_boxes})

        if len(TPS) > 0:

            tp_boxes = np.array([[tp[0], tp[1], tp[0]+tp[2], tp[1]+tp[3]] for tp in TPS])
            tp_scores = np.array([tp[4] for tp in TPS])

            fp_boxes = np.array([[fp[0], fp[1], fp[0]+fp[2], fp[1]+fp[3]] for fp in FPS])
            fp_scores = np.array([fp[4] for fp in FPS])

            fn_boxes = np.array([[fn[0], fn[1], fn[0]+fn[2], fn[1]+fn[3]] for fn in FNS])
            
            images2inspect_TPS.append({"id": id, "fn_boxes": fn_boxes, "tp_boxes": tp_boxes, "fp_boxes": fp_boxes})

        all_images.append(d_img)

    if inspect_images:
        return images2inspect_TPS, images2inspect_FPS, images2inspect_FNS
    else:
        return all_images
    
def decision_strategy_1(dataset, im_id: int, Dt: COCO, Dt_parts: COCO, score_thresh: float, score_thresh_parts: List[float], overlap_threshold=0.01):
    # per-image evaluation
    
    img = dataset.loadImgs(im_id)[0]

    dets_person_ids = Dt.getAnnIds(imgIds=img['id'])
    detections_person = Dt.loadAnns(dets_person_ids)

    boxes_person = np.array([det['bbox'] for det in detections_person])
    scores_person = np.array([det['score'] for det in detections_person])
    labels_person = np.array([det['category_id'] for det in detections_person])

    keep = scores_person >= score_thresh
    labels_person = labels_person[keep]
    scores_person = scores_person[keep]
    boxes_person = boxes_person[keep]

    dets_parts_ids = Dt_parts.getAnnIds(imgIds=img['id'])
    detections_parts = Dt_parts.loadAnns(dets_parts_ids)

    boxes_parts = np.array([det['bbox'] for det in detections_parts])
    scores_parts = np.array([det['score'] for det in detections_parts])
    labels_parts = np.array([det['category_id'] for det in detections_parts])

    keep = discard_part_detection(labels=labels_parts, scores=scores_parts, score_thresholds_parts=score_thresh_parts)
    labels_parts = labels_parts[keep]
    scores_parts = scores_parts[keep]
    boxes_parts = boxes_parts[keep]

    # looking for False Positives
    FP_detected=False
    for i_person in range(len(scores_person)):
        box_person = boxes_person[i_person, :]
        detected = True
        for i_part in range(len(scores_parts)):
            overlap = bbox_intersection(boxes_parts[i_part, :], box_person, x1y1x2y2=False)
            area_part = bbox_area(boxes_parts[i_part, :], x1y1x2y2=False)
            area_person = bbox_area(box_person, x1y1x2y2=False)
            area = min(area_part, area_person)
            if overlap >= (1.-overlap_threshold)*area:
                detected = False
                break
        if detected:
            FP_detected=True
            break
        
    # looking for False Negatives   
    FN_detected=False
    for i_part in range(len(scores_parts)):
        detected = True
        for i_person in range(len(scores_person)):
            overlap = bbox_intersection(boxes_parts[i_part, :], boxes_person[i_person, :], x1y1x2y2=False)
            area_part = bbox_area(boxes_parts[i_part, :], x1y1x2y2=False)
            area_person = bbox_area(box_person, x1y1x2y2=False)
            area = min(area_part, area_person)
            if overlap >= (1.-overlap_threshold)*area:
                detected = False
                break
        if detected:
            FN_detected=True
            break
    
    return FP_detected, FN_detected

def decision_strategy_2(dataset, im_id: int, Dt: COCO, Dt_parts: COCO, score_thresh: float, score_thresh_parts: List[float], overlap_threshold=0.01):
    # per-object evaluation
    
    img = dataset.loadImgs(im_id)[0]

    dets_person_ids = Dt.getAnnIds(imgIds=img['id'])
    detections_person = Dt.loadAnns(dets_person_ids)

    boxes_person = np.array([det['bbox'] for det in detections_person])
    scores_person = np.array([det['score'] for det in detections_person])
    labels_person = np.array([det['category_id'] for det in detections_person])

    keep = scores_person >= score_thresh
    labels_person = labels_person[keep]
    scores_person = scores_person[keep]
    boxes_person = boxes_person[keep]

    dets_parts_ids = Dt_parts.getAnnIds(imgIds=img['id'])
    detections_parts = Dt_parts.loadAnns(dets_parts_ids)

    boxes_parts = np.array([det['bbox'] for det in detections_parts])
    scores_parts = np.array([det['score'] for det in detections_parts])
    labels_parts = np.array([det['category_id'] for det in detections_parts])

    keep = discard_part_detection(labels=labels_parts, scores=scores_parts, score_thresholds_parts=score_thresh_parts)
    labels_parts = labels_parts[keep]
    scores_parts = scores_parts[keep]
    boxes_parts = boxes_parts[keep]
    
    l_TP = []
    l_FP = []
    l_FN = []
    
    # looking for False Positives
    for i_person in range(len(scores_person)):
        box_person = boxes_person[i_person, :]
        is_TP = False
        for i_part in range(len(scores_parts)):
            overlap = bbox_intersection(boxes_parts[i_part, :], box_person, x1y1x2y2=False)
            area_part = bbox_area(boxes_parts[i_part, :], x1y1x2y2=False)
            area_person = bbox_area(box_person, x1y1x2y2=False)
            area = min(area_part, area_person)
            if overlap >= (1.-overlap_threshold)*area:
                box_person = box_person.tolist()
                l_TP.append([box_person[0], box_person[1], box_person[2], box_person[3]])
                is_TP = True
                break
        if not is_TP:
            box_person = box_person.tolist()
            l_FP.append([box_person[0], box_person[1], box_person[2], box_person[3]])
            
    # looking for False Negatives   
    for i_part in range(len(scores_parts)):
        is_FN = True
        box_part = boxes_parts[i_part, :]
        for i_person in range(len(scores_person)):
            overlap = bbox_intersection(box_part, boxes_person[i_person, :], x1y1x2y2=False)
            area_part = bbox_area(box_part, x1y1x2y2=False)
            area_person = bbox_area(boxes_person[i_person, :], x1y1x2y2=False)
            area = min(area_part, area_person)
            if overlap >= (1.-overlap_threshold)*area:
                is_FN = False
                break
        if is_FN:
            box_part = box_part.tolist()
            l_FN.append([box_part[0], box_part[1], box_part[2], box_part[3]])

    return l_TP, l_FP, l_FN