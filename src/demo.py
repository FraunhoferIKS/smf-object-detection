import argparse
import cv2
from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN

import torch.nn as nn
from mmcv import Config
from mmdet.apis import inference_detector, init_detector

from src.tools.utils import bbox_area, compute_center, bbox_intersection, discard_part_detection, get_minimal_enclosing_bbox
from src.tools.visualization import draw_bounding_boxes

def smf(detections: Dict) -> Dict:
    """Self-Monitoring Framework: Monitor that takes a dictionary with person and part detections and groups them into potential TP, FP, FN detections

    Args:
        detections (Dict): Person (boxes + scores) and body-parts (boxes, scores, labels) detections 

    Returns:
        Dict: Predicted True Positives, False Positive, and False Negative detections
    """

    OVERLAP_THRESHOLD_FN = 0.99
    OVERLAP_THRESHOLD_FP = 0.1
    
    bbox_person = detections['person']['boxes']
    scores_person = detections['person']['scores']
    bbox_parts = detections['parts']['boxes']
    scores_parts = detections['parts']['scores']
    labels_parts = detections['parts']['labels']
    
    if bbox_person is None:
        bbox_person = np.empty(shape=(0, 4))
    if bbox_parts is None:
        bbox_parts = np.empty(shape=(0, 4))
        scores_parts = np.empty(shape=(0,))
        labels_parts = np.empty(shape=(0,))
    
    d_TP, d_FP, d_FN = {"boxes": [], "scores": [], "labels": []}, {"boxes": [], "scores": [], "labels": []}, {"boxes": [], "scores": [], "labels": []}

    # looking for False Positives
    for i_person in range(bbox_person.shape[0]):
        b_person = bbox_person[i_person, :]
        is_TP = False

        for i_part in range(bbox_parts.shape[0]):
            overlap = bbox_intersection(bbox_parts[i_part, :], b_person, x1y1x2y2=False)
            area_part = bbox_area(bbox_parts[i_part, :], x1y1x2y2=False)
            area_person = bbox_area(b_person, x1y1x2y2=False)
            area = min(area_part, area_person)

            if overlap >= (1.-OVERLAP_THRESHOLD_FP)*area:
                b_person = b_person.tolist()
                d_TP["boxes"].append([b_person[0], b_person[1], b_person[2], b_person[3]])
                d_TP["scores"].append(scores_person[i_person])
                d_TP["labels"].append(0)
                is_TP = True
                break

        if not is_TP:
            b_person = b_person.tolist()
            d_FP["boxes"].append([b_person[0], b_person[1], b_person[2], b_person[3]])
            d_FP["scores"].append(scores_person[i_person])
            d_FP["labels"].append(0)
            
    # looking for False Negatives   
    for i_part in range(bbox_parts.shape[0]):
        is_FN = True
        b_part = bbox_parts[i_part, :]

        for i_person in range(bbox_person.shape[0]):
            overlap = bbox_intersection(b_part, bbox_person[i_person, :], x1y1x2y2=False)
            area_part = bbox_area(b_part, x1y1x2y2=False)
            area_person = bbox_area(bbox_person[i_person, :], x1y1x2y2=False)
            area = min(area_part, area_person)

            if overlap >= (1.-OVERLAP_THRESHOLD_FN)*area:
                is_FN = False
                break

        if is_FN:
            b_part = b_part.tolist()
            d_FN["boxes"].append([b_part[0], b_part[1], b_part[2], b_part[3]])
            d_FN["scores"].append(scores_parts[i_part])
            d_FN["labels"].append(labels_parts[i_part])
    
    d_TP["boxes"], d_TP["scores"], d_TP["labels"] = np.array(d_TP["boxes"]), np.array(d_TP["scores"]), np.array(d_TP["labels"])
    d_FP["boxes"], d_FP["scores"], d_FP["labels"] = np.array(d_FP["boxes"]), np.array(d_FP["scores"]), np.array(d_FP["labels"])       
    d_FN["boxes"], d_FN["scores"], d_FN["labels"] = np.array(d_FN["boxes"]), np.array(d_FN["scores"]), np.array(d_FN["labels"])
            
    return {"TPs": d_TP, "FPs": d_FP, "FNs": d_FN}

def process_detections(detections: List[np.array], score_threshold_person: float=None, score_thresholds_parts: List[float]=None) -> Dict:   
    """Takes detections from mmdetection model as input and outputs detections in new format,to make it also compatible with multilabel model

    Args:
        detections (List[np.array]): person/body-parts detections
        score_threshold_person (float, optional): Score threshold for class person. Defaults to None.
        score_thresholds_parts (List[float], optional): per-class score threshold for body-parts. Defaults to None.

    Returns:
        Dict: output detections in new format
    """
    
    scores_person, scores_parts = None, None
    bboxes_person, bboxes_parts = None, None
    labels_parts = None
    
    if score_threshold_person is not None:
    
        dets_person = detections[0]
            
        if len(dets_person.shape) == 1:
            dets_person = np.expand_dims(dets_person, axis=0)
                
        scores_person = dets_person[:, -1]
        bboxes_person = dets_person[:, :-1]
            
        keep = scores_person >= score_threshold_person
            
        scores_person = scores_person[keep]
        bboxes_person = bboxes_person[keep]
        
    if score_thresholds_parts is not None:
            
        dets_parts = [detections[c] for c in range(1, len(detections))]
        if len(dets_parts):
            dets_parts = np.vstack(dets_parts)
        else:
            dets_parts = np.empty(shape=(0, 4))
            
        if len(dets_parts.shape) == 1:
            dets_parts = np.expand_dims(dets_parts, axis=0)
            
        scores_parts, bboxes_parts, labels_parts = [], [], []
            
        for i, bbox in enumerate(detections):
            if i==0:
                continue
                
            if len(bbox.shape) == 1:
                bbox = np.expand_dims(bbox, axis=0) 
                    
            labels_parts.extend([i]*bbox.shape[0])
            scores_parts.extend(bbox[:, -1])
            bboxes_parts.extend(bbox[:, :-1].tolist())
            
        labels_parts = np.array(labels_parts)
        scores_parts = np.array(scores_parts)
        bboxes_parts = np.array(bboxes_parts)
            
        if len(labels_parts):
                
            keep = discard_part_detection(labels=labels_parts, scores=scores_parts, score_thresholds_parts=score_thresholds_parts)

            labels_parts = labels_parts[keep]
            scores_parts = scores_parts[keep]
            bboxes_parts = bboxes_parts[keep]

    return {"person": {"boxes": bboxes_person, "scores": scores_person}, "parts": {"boxes": bboxes_parts, "scores": scores_parts, "labels": labels_parts}}

def fuse_unmatched_parts(FNS: np.array) -> Tuple[np.array, np.array]:
    """Clusters unmatched body-parts detections and for each cluster, a new enclosing bbox is calculated with a confidence score averaged over the scores of the cluster

    Args:
        FNS (np.array): Unmatched body-parts detections (potential False Negatives)

    Returns:
        Tuple[np.array, np.array]: enclosing bboxes with average scores
    """

    # compute center points
    l_centers = np.array([compute_center(FNS[i, :-1].tolist(), x1y1x2y2=True) for i in range(FNS.shape[0])])
    # cluster points
    dbscan = DBSCAN(eps=500, # maximum distance between two samples for one to be considered as in the neighbourhood of the other
                    min_samples=1, # the number of samples in a neighbourhood for a point to be considered as a core point
                    metric='euclidean'
                    )
    # Fit the DBSCAN object to the data
    dbscan.fit(l_centers)
    # Get the cluster labels
    c_labels = dbscan.labels_ # cluster labels for each point in the dataset given to fit(), noisy samples are given the label -1
    valid = c_labels >= 0
    
    boxes = np.empty(shape=(0, 4))
    mean_scores = np.empty(shape=(0,))
    
    if np.any(valid == True):
        
        s_c_labels = set(c_labels)
        
        for i in s_c_labels:
            if i < 0:
                continue
            keep = c_labels == i
            c_boxes = FNS[:, :-1][keep]
            if c_boxes.shape[0] > 1: # only use cluster with more than 1 point to reduce ghost body-parts
                # enclosing_score = np.expand_dims(np.expand_dims(FNS[:, -1][keep], axis=1).mean(0), axis=0)
                enclosing_score = np.expand_dims(FNS[:, -1][keep], axis=1).mean(0)
                enclosing_box = np.expand_dims(get_minimal_enclosing_bbox(boxes=c_boxes), axis=0)
                boxes = np.concatenate((boxes, enclosing_box), axis=0)
                mean_scores = np.concatenate((mean_scores, enclosing_score), axis=0)
            
    return boxes, mean_scores

def demo(args, model_person: nn.Module, model_parts: nn.Module, img: np.array) -> np.array:
    """Runs the demo on a single image and outputs an image with respective detections

    Returns:
        np.array: image with detections
    """
    
    # bbox settings
    bbox_alpha=0.5
    bbox_is_opaque=True
    bbox_top=True
    
    detections_person = inference_detector(model_person, img)
    detections_processed_person = process_detections(detections_person, score_threshold_person=args.score_thresh_person)
    
    detections_parts= inference_detector(model_parts, img)
    detections_processed_parts = process_detections(detections_parts, score_thresholds_parts=args.score_thresh_parts)
    
    detections_processed = detections_processed_person
    detections_processed["parts"] = detections_processed_parts["parts"]
    
    checked_output = smf(detections=detections_processed)
    
    if len(checked_output["TPs"]["scores"]):

        img = draw_bounding_boxes(
            img, 
            boxes=checked_output["TPs"]["boxes"], 
            scores=checked_output["TPs"]["scores"],
            labels=np.zeros_like(checked_output["TPs"]["scores"]),
            mapping={0: "TP"},
            color=(0, 255, 0),
            is_opaque=bbox_is_opaque,
            alpha=bbox_alpha,
            top=bbox_top
            )
        
    if len(checked_output["FPs"]["scores"]):

        img = draw_bounding_boxes(
            img, 
            boxes=checked_output["FPs"]["boxes"], 
            scores=checked_output["FPs"]["scores"],
            labels=np.zeros_like(checked_output["FPs"]["scores"]),
            mapping={0: "FP"},
            color=(255, 0, 0),
            is_opaque=bbox_is_opaque,
            alpha=bbox_alpha,
            top=bbox_top
            )
        
    if len(checked_output["FNs"]["scores"]):
                    
            FNS = np.concatenate((checked_output["FNs"]["boxes"], np.expand_dims(checked_output["FNs"]["scores"], axis=1)), axis=1) 
            enclosing_boxes, enclosing_scores = fuse_unmatched_parts(FNS)
                
            if enclosing_boxes.shape[0] > 0:
                img = draw_bounding_boxes(
                    img, 
                    boxes=enclosing_boxes, 
                    scores=enclosing_scores,
                    labels=np.zeros_like(enclosing_scores),
                    mapping={0: "FN"},
                    color=(0, 0, 255),
                    is_opaque=bbox_is_opaque,
                    alpha=bbox_alpha,
                    top=bbox_top
                )
    
    return img

def main():
    
    args = parse_args()
    
    cfg_person = Config.fromfile(args.config_person)
    cfg_parts = Config.fromfile(args.config_parts)
    
    # initialize object detectors
    model_person = init_detector(cfg_person, args.checkpoint_person, device=args.device)
    model_parts = init_detector(cfg_parts, args.checkpoint_parts, device=args.device)
    
    args.score_thresh_person = float(args.score_thresh_person)
    args.score_thresh_parts = [float(score_thresh) for score_thresh in args.score_thresh_parts]
    
    img = cv2.imread(args.image_path)
    
    img = demo(
        args=args, 
        model_person=model_person, 
        model_parts=model_parts,
        img=img
        )
    
    if args.out is not None:
        cv2.imwrite(args.out, img)
    
def parse_args():
    
    parser = argparse.ArgumentParser(description='MMDetection images demo')
    parser.add_argument('--image-path', help='Image Folder', type=str, 
                        default=None)
    parser.add_argument('--config-person', help='Config file', type=str,
                        default=None)
    parser.add_argument('--checkpoint-person', type=str, help='Checkpoint file', 
                        default=None)
    parser.add_argument('--config-parts', help='Config file', type=str, 
                        default=None)
    parser.add_argument('--checkpoint-parts', type=str, help='Checkpoint file', 
                        default=None)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score-thresh-person', 
                        default=None)
    parser.add_argument('--score-thresh-parts', nargs="+", 
                        default=None)
    parser.add_argument('--out', type=str, help='Output image file path', 
                        default=None)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    main()