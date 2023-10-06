import numpy as np
from typing import List
import pycocotools.mask as mask_util

map_multilabel = {
    1: "Person",
    2: "Torso",
    3: "Hand",
    4: "Foot",
    5: "Upper Leg",
    6: "Lower Leg", 
    7: "Upper Arm",
    8: "Lower Arm",
    9: "Head"
}

map_reduced_DensePose = {
    1: "Torso",
    2: "Hand",
    3: "Foot",
    4: "Upper Leg",
    5: "Lower Leg",
    6: "Upper Arm",
    7: "Lower Arm",
    8: "Head"
}

map_DensePose = {
    1: "Torso",
    2: "Right Hand",
    3: "Left Hand",
    4: "Right Foot",
    5: "Left Foot",
    6: "Upper Leg Right",
    7: "Upper Leg Left",
    8: "Lower Leg Right",
    9: "Lower Leg Left",
    10: "Upper Arm Left",
    11: "Upper Arm Right",
    12: "Lower Arm Left",
    13: "Lower Arm Right",
    14: "Head"
}

map_person = {
    1: "Person"
}
    
def discard_part_detection(labels: np.array, scores: np.array, score_thresholds_parts: List[float]) -> List[bool]:
    per_class_thresh = [score_thresholds_parts[l-1] for l in labels.tolist()]
    keep = [True if score >= thresh else False for (score, thresh) in zip(scores, per_class_thresh)]
    return keep

def compute_center(box, x1y1x2y2=True) -> List:

    if x1y1x2y2:
        c_x = box[0]+0.5*(box[2]-box[0])
        c_y = box[1]+0.5*(box[3]-box[1])
    else:
        c_x = box[0]+0.5*box[2]
        c_y = box[1]+0.5*box[3]

    return [c_x, c_y]

def bbox_area(box, x1y1x2y2=True):
    if x1y1x2y2:
        area = (box[2]-box[0])*(box[3]-box[1])
    else:
        area = box[2]*box[3]
    return area

def bbox_iou(box1, box2, x1y1x2y2=True):

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea / uarea

def bbox_intersection(box1, box2, x1y1x2y2=True):

    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]

        mx = min(box1[0], box2[0])
        Mx = max(box1[0] + w1, box2[0] + w2)
        my = min(box1[1], box2[1])
        My = max(box1[1] + h1, box2[1] + h2)
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0
    carea = cw * ch
    return carea

def GetDensePoseMask(Polys):
    MaskGen = np.zeros([256,256])
    for i in range(1,15):
        if(Polys[i-1]):
            current_mask = mask_util.decode(Polys[i-1])
            MaskGen[current_mask>0] = i
    return MaskGen

def get_minimal_bbox(Mask: np.array, body_part: int):

    bx1, bx2, by1, by2, = None, None, None, None
    
    for r in range(Mask.shape[0]):
        if any(Mask[r, :] > 0.):
            bx1 = r
            break
    for r in range(Mask.shape[0]-1, -1, -1):
            if any(Mask[r, :] > 0.):
                bx2 = r
                break
    for c in range(Mask.shape[1]):
        if any(Mask[:, c] > 0.):
            by1 = c
            break
    for c in range(Mask.shape[1]-1, -1, -1):
        if any(Mask[:, c] > 0.):
            by2 = c
            break
        
    for r in range(Mask.shape[0]):
        for c in range(Mask.shape[1]):
            if r >= bx1 and r <= bx2 and by1 <= c and by2 >= c:
                Mask[r, c] = body_part

    return Mask

def get_minimal_enclosing_bbox(boxes: np.array, x1y1x2y2: bool=True) -> List:
    """Takes a numpy array of bboxes and computes a new bbox box that encloses all bboxes

    Args:
        boxes (np.array): bboxes of detections
        x1y1x2y2 (bool, optional): Bbox format. Defaults to True.

    Returns:
        List: Minimal bbox that encloses all given bboxes
    """
    
    if not x1y1x2y2:
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
    
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i, :]
        min_x = min(min_x, x1)
        min_y = min(min_y, y1)
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)
        
    width = max_x - min_x
    height = max_y - min_y
    
    if x1y1x2y2:
        enclosing_box = [min_x, min_y, max_x, max_y]
    else: 
        enclosing_box = [min_x, min_y, width, height]
    return enclosing_box
