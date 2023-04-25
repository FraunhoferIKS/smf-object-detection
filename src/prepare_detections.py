import os
import json
import argparse
from pathlib import Path
from pycocotools.coco import COCO

def create_person_part_detections_from_baseline(path_anns: str, path_dets_person: str, path_dets_baseline: str, save_path_person: str, save_path_parts: str):
    
    dataset = COCO(path_anns)
    im_ids = dataset.getImgIds()
    
    Dt = dataset.loadRes(path_dets_person)
    Dt_baseline = dataset.loadRes(path_dets_baseline)
    
    coco_person_results = []
    coco_parts_results = []
    
    for id in im_ids:
        
        img = dataset.loadImgs(id)[0]
        dets_ids_person = Dt.getAnnIds(imgIds=img['id'])
        detections_person = Dt.loadAnns(dets_ids_person)
        dets_ids_baseline = Dt_baseline.getAnnIds(imgIds=img['id'])
        detections_baseline = Dt_baseline.loadAnns(dets_ids_baseline)
        
        dets_person = [det for det in detections_person]
        dets_parts = [det for det in detections_baseline]
        
        coco_person_results.extend(
                        [
                            {
                                "image_id": id,
                                "category_id": det['category_id'],
                                "bbox": det['bbox'],
                                "score": det['score'],
                            }
                            for det in dets_person
                        ]
                    )

        coco_parts_results.extend(
                        [
                            {
                                "image_id": id,
                                "category_id": det['category_id'],
                                "bbox": det['bbox'],
                                "score": det['score'],
                            }
                            for det in dets_parts
                        ]
                    )
        
    json.dump(coco_person_results, open(save_path_person, 'w'), indent=1)
    json.dump(coco_parts_results, open(save_path_parts, 'w'), indent=1)

    print('Detections saved.')

def create_person_part_detections_from_multilabel(path_anns: str, path_dets_multilabel: str, save_path_person: str, save_path_parts: str):
    
    dataset = COCO(path_anns)
    im_ids = dataset.getImgIds()
    Dt = dataset.loadRes(path_dets_multilabel)
    
    coco_person_results = []
    coco_parts_results = []
    
    for id in im_ids:
        
        img = dataset.loadImgs(id)[0]
        dets_ids = Dt.getAnnIds(imgIds=img['id'])
        detections_multilabel = Dt.loadAnns(dets_ids)
        
        dets_person = [det for det in detections_multilabel if det["category_id"] == 1]
        dets_parts = [det for det in detections_multilabel if det["category_id"] != 1]
        
        coco_person_results.extend(
                        [
                            {
                                "image_id": id,
                                "category_id": det['category_id'],
                                "bbox": det['bbox'],
                                "score": det['score'],
                            }
                            for det in dets_person
                        ]
                    )

        coco_parts_results.extend(
                        [
                            {
                                "image_id": id,
                                "category_id": det['category_id'],
                                "bbox": det['bbox'],
                                "score": det['score'],
                            }
                            for det in dets_parts
                        ]
                    )
        
    json.dump(coco_person_results, open(save_path_person, 'w'), indent=1)
    json.dump(coco_parts_results, open(save_path_parts, 'w'), indent=1)

    print('Detections saved.')
    
if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Conversion')
    
    args.add_argument('--path_anns', type=str)

    args.add_argument('--path_dir_multilabel', type=str, default=None)
    
    args.add_argument('--path_dir_person', type=str, default=None)
    
    args.add_argument('--path_dir_baseline', type=str, default=None)

    args.add_argument('--json_file', type=str)
    
    args = args.parse_args()
    
    if args.path_dir_multilabel is not None and args.path_dir_person is None and args.path_dir_baseline is None:
        
        Path(os.path.join(args.path_dir_multilabel, "coco_person")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.path_dir_multilabel, "coco_parts")).mkdir(parents=True, exist_ok=True)
    
        path_dets_multilabel = os.path.join(args.path_dir_multilabel, args.json_file)
        save_path_person = os.path.join(args.path_dir_multilabel, "coco_person", args.json_file)
        save_path_parts = os.path.join(args.path_dir_multilabel, "coco_parts", args.json_file)
    
        create_person_part_detections_from_multilabel(args.path_anns, path_dets_multilabel, save_path_person, save_path_parts)
        
    elif args.path_dir_multilabel is None and args.path_dir_person is not None and args.path_dir_baseline is not None:
        
        Path(os.path.join(args.path_dir_baseline, "coco_person")).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(args.path_dir_baseline, "coco_parts")).mkdir(parents=True, exist_ok=True)
        
        path_dets_person = os.path.join(args.path_dir_person, args.json_file)
        path_dets_baseline = os.path.join(args.path_dir_baseline, args.json_file)
        
        save_path_person = os.path.join(args.path_dir_baseline, "coco_person", args.json_file)
        save_path_parts = os.path.join(args.path_dir_baseline, "coco_parts", args.json_file)
        
        create_person_part_detections_from_baseline(path_anns=args.path_anns, path_dets_person=path_dets_person, path_dets_baseline=path_dets_baseline, 
                                                    save_path_person=save_path_person, save_path_parts=save_path_parts)
    else:
        raise ValueError("Combinations of arguments is invalid.")