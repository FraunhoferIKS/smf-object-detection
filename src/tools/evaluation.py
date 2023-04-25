import numpy as np
from typing import Tuple, List, Dict, Any
from torch import Tensor

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

def accumulate(
    eval_images: List[Dict[str, Any]], cat_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert eval images for every category into a mask for detections and annotations
    Copied and modified from https://github.com/cocodataset/cocoapi/blob/6c3b394c07aed33fd83784a8bf8798059a1e9ae4/PythonAPI/pycocotools/cocoeval.py#L315
    Args:
        eval_images: Evaluation results from pycocotools
        cat_ids: List of category (label) ids
        n_gt: Number of annotations
        n_det: Number of detections
    Returns:
        False positive binary mask
        False negative binary mask
    """

    fp_mask = []
    fn_mask = []
    dtIds = []
    gtIds = []

    for k, _ in enumerate(cat_ids):
        if eval_images[k] is None:
            continue

        dtm = eval_images[k]["dtMatches"]
        gtm = eval_images[k]["gtMatches"]
        dtIg = eval_images[k]["dtIgnore"]
        gtIg = eval_images[k]["gtIgnore"]

        dtIds.append(eval_images[k]["dtIds"])
        gtIds.append(eval_images[k]["gtIds"])

        if len(eval_images[k]["dtIds"]) > 0:
            fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
            # fp_mask[k, :] = fps.squeeze()
            fp_mask.append(fps.squeeze())
        if len(eval_images[k]["gtIds"]) > 0:
            fns = np.logical_and(np.logical_not(gtm), np.logical_not(gtIg))
            fn_mask.append(fns.squeeze())
            # fn_mask[k, :] = fns.squeeze()
    return np.array(fp_mask), np.array(fn_mask), dtIds, gtIds

def get_cocoeval_error_masks(dataset, cocoDt: COCO, im_id: int, cat_ids: List[int], 
        iou_threshold: float, area_rng: List[float]=[0.0, 10000000000.0]):

    max_dets=100
    
    coco_eval = COCOeval(cocoGt=dataset, cocoDt=cocoDt, iouType="bbox")
    coco_eval.params.maxDets = [max_dets]
    coco_eval.params.iouThrs = [iou_threshold]
    coco_eval.params.imgIds = [im_id]
    coco_eval.params.areaRng = [area_rng]
    
    coco_eval.evaluate()
    eval_imgs = coco_eval.evalImgs
    fp_mask, fn_mask, dtIds, gtIds = accumulate(eval_imgs, cat_ids)
    return fp_mask, fn_mask, dtIds, gtIds

def evaluate(path_to_anns: str, path_to_dets: str, cats: List[int]=None):
    
    coco = COCO(path_to_anns)
    cocoDt = coco.loadRes(path_to_dets)

    coco_evaluator = COCOeval(cocoGt=coco, cocoDt=cocoDt, iouType="bbox")

    if cats is not None:
        coco_evaluator.params.catIds = cats

    # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    coco_evaluator.evaluate()

    # Accumulate per image evaluation results and store the result in self.eval
    coco_evaluator.accumulate()

    # Compute and display summary metrics for evaluation results.
    # Note this functin can *only* be applied on the default parameter setting
    coco_evaluator.summarize()

    return coco_evaluator.stats

def compute_metrics(path_to_anns: str, path_to_dets: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]: 

    coco = COCO(path_to_anns)
    cocoDt = coco.loadRes(path_to_dets)

    coco_evaluator = COCOeval(cocoGt=coco, cocoDt=cocoDt, iouType="bbox")

    # if cats is not None:
    #     coco_evaluator.params.catIds = cats

    # Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    coco_evaluator.evaluate()

    # Accumulate per image evaluation results and store the result in self.eval
    coco_evaluator.accumulate()
    
    # [T, R, K, A, M]
    # K: Categories
    # R: Recall values
    # T: IoU thresholds (0: average over all IoU, 1: IoU@0.5), for person, IoU=0.5 is often used
    # A: Object area ranges for evaluation (A=4), areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
    # M: [1, 10, 100] (M=3)
    
    T = 1 # IoU@0.5
    M = 2
    A = 0
    
    precision = coco_evaluator.eval['precision'][T, :, :, A, M]
    scores = coco_evaluator.eval['scores'][T, :, :, A, M]
    recall =  np.arange(0., 1.01, 0.01)

    f1_score = 2.* np.divide(np.multiply(precision, np.expand_dims(recall, axis=-1)), precision+np.expand_dims(recall, axis=-1))

    return scores, precision, recall, f1_score
