import argparse
import os, json
import cv2

import numpy as np
from pycocotools.coco import COCO
from src.tools.utils import GetDensePoseMask, get_minimal_bbox, map_DensePose, map_person, map_multilabel, map_reduced_DensePose

MIN_AREA = 2247. # The minimal are of person bboxes the detectors have seen during training

# *person: n_classes: 1 ('Person')
# *parts: n_classes: 8 ("Torso", "Hand", "Foot", "Upper Leg", "Lower Leg", "Upper Arm", "Lower Arm", "Head")
# *multilabel: n_classes: 9 ("Person", "Torso", "Hand", "Foot", "Upper Leg", "Lower Leg", "Upper Arm", "Lower Arm", "Head")

def use_annotations_voc2010(args, new_anns):
    
    voc2010 = COCO(args.old_anns)
    im_ids = voc2010.getImgIds()

    # Load annotations
    for im_id in im_ids:
        
        person_present=False

        im = voc2010.loadImgs(im_id)[0]

        ann_ids = voc2010.getAnnIds(imgIds=im['id'])
        anns = voc2010.loadAnns(ann_ids)
        
        tmp_anns = []
        too_small_bbox = False
        for ann in anns: 
            if ann['category_id'] == 15:
                person_present = True
                new_ann = {"area": ann["area"], "iscrowd": ann['iscrowd'], "image_id": im_id, "bbox": ann['bbox'], "category_id": 1, 
                           "id": ann["id"]}
                if ann['area'] < MIN_AREA:
                    too_small_bbox = True
                    break
                tmp_anns.append(new_ann)
                
        if too_small_bbox:
            continue
        new_anns["annotations"].extend(tmp_anns)   
          
        if person_present:
            new_anns["images"].append(im)
    
    print(f'Dataset size: {len(new_anns["images"])}')
    with open(os.path.join(args.new_anns_dir, args.new_anns_file), "w") as out:
        json.dump(new_anns, out, indent=4)
    print("Annotations created.")

def use_multilabel_coco2014_dataset(args, new_anns):

    dp_coco = COCO(args.old_anns)
    im_ids = dp_coco.getImgIds()

    # only keep images where all persons have GT body-part segmentation masks
    percentage_bp_ann = []
    for id in im_ids:
        img = dp_coco.loadImgs(id)[0]
        ann_ids = dp_coco.getAnnIds(imgIds=img['id'])
        annotations = dp_coco.loadAnns(ann_ids)
        dp_ann = 0
        bbox_ann = 0
        for ann in annotations:
            if ann["category_id"] != 1:
                continue
            bbox_ann += 1
            if('dp_masks' in ann.keys()):
                dp_ann += 1

        if bbox_ann > 0:
            percentage_bp_ann.append(dp_ann/bbox_ann)
        
    np_percentage_bp_ann = np.array(percentage_bp_ann)
    keep = np_percentage_bp_ann == 1.
    im_ids = np.array(im_ids)[keep].tolist()

    # Load annotations
    for im_id in im_ids:

        im = dp_coco.loadImgs(im_id)[0]
        new_anns["images"].append(im)

        ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
        anns = dp_coco.loadAnns(ann_ids)

        n_ids = []

        for ann in anns: 

            width, height = im["width"], im["height"]
            bbr =  np.array(ann['bbox']).astype(int)
            x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]

            Mask = GetDensePoseMask(ann['dp_masks'])

            new_ann = {"area": ann["area"], "iscrowd": ann['iscrowd'], "image_id": im_id, "bbox": ann['bbox'], "category_id": 1, 
                       "id": ann["id"]}
            new_anns["annotations"].append(new_ann)

            for body_part in map_DensePose.keys():

                if body_part not in Mask:
                    continue

                bp_mask = Mask.copy()
                bp_mask[bp_mask != body_part] = 0. # only body part mask is considered
                bp_mask = get_minimal_bbox(bp_mask, body_part)
                bp_maskIm = cv2.resize(bp_mask, (int(x2-x1),int(y2-y1)), interpolation=cv2.INTER_NEAREST)

                y, x = np.where(bp_maskIm > 0.)
                bp_x1, bp_y1, bp_x2, bp_y2 =  int(np.min(x)+x1), int(np.min(y)+y1), int(np.max(x)+x1), int(np.max(y)+y1)
                box = [bp_x1, bp_y1, bp_x2-bp_x1, bp_y2-bp_y1]

                if box[2] <= 0 or box[3] <= 0:
                    del bp_mask
                    continue

                assert(box[0] < width and box[0] >= 0)
                assert(box[1] < height and box[1] >= 0)
                assert(box[0] + box[2] < width)
                assert(box[1] + box[3] < height)
                assert(box[2] > 0 and box[3] > 0) 

                # create new annotation
                new_id = int(str(ann["id"]) + str(00) + str(body_part))
                if new_id in ann_ids or new_id in n_ids:
                    raise ValueError("id already exists.")
                n_ids.append(new_id)

                if body_part == 1:
                    body_part = 2
                elif body_part == 2:
                    body_part = 3
                elif body_part == 6 or body_part == 7:
                    body_part = 5
                elif body_part == 8 or body_part == 9:
                    body_part = 6
                elif body_part == 10 or body_part == 11:
                    body_part = 7
                elif body_part == 12 or body_part == 13:
                    body_part = 8
                elif body_part == 14:
                    body_part = 9

                new_ann = {"area": box[2]*box[3], "iscrowd": 0, "image_id": im_id, "bbox": box, "category_id": body_part, "id": new_id}
                new_anns["annotations"].append(new_ann)

                del bp_mask

    with open(os.path.join(args.new_anns_dir, args.new_anns_file), "w") as out:
        json.dump(new_anns, out, indent=4)
    print("Annotations created.")
    
def use_densePose_dataset(args, new_anns):

    dp_coco = COCO(args.old_anns)
    im_ids = dp_coco.getImgIds()

    # only keep images where all persons have GT body-part segmentation masks
    percentage_bp_ann = []
    for id in im_ids:
        img = dp_coco.loadImgs(id)[0]
        ann_ids = dp_coco.getAnnIds(imgIds=img['id'])
        annotations = dp_coco.loadAnns(ann_ids)
        dp_ann = 0
        bbox_ann = 0
        for ann in annotations:
            if ann["category_id"] != 1:
                continue
            bbox_ann += 1
            if('dp_masks' in ann.keys()):
                dp_ann += 1

        if bbox_ann > 0:
            percentage_bp_ann.append(dp_ann/bbox_ann)
        
    np_percentage_bp_ann = np.array(percentage_bp_ann)
    keep = np_percentage_bp_ann == 1.
    im_ids = np.array(im_ids)[keep].tolist()
    
    l_arng = []
    
    print(f'Number of samples: {len(im_ids)}')

    # Load annotations
    for im_id in im_ids:

        im = dp_coco.loadImgs(im_id)[0]
        new_anns["images"].append(im)

        ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
        anns = dp_coco.loadAnns(ann_ids)

        n_ids = []

        for ann in anns: 
            
            if 'dp_masks' in ann.keys():

                width, height = im["width"], im["height"]
                bbr =  np.array(ann['bbox']).astype(int)
                x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]

                Mask = GetDensePoseMask(ann['dp_masks'])

                if args.anns_mode == "person":
                    new_ann = {"area": ann["area"], "iscrowd": ann['iscrowd'], "image_id": im_id, "bbox": ann['bbox'], "category_id": 1, 
                               "id": ann["id"]}
                    new_anns["annotations"].append(new_ann)
                    l_arng.append(ann["area"])

                elif args.anns_mode == "parts":
                    
                    for body_part in map_DensePose.keys():

                        if body_part not in Mask:
                            continue

                        bp_mask = Mask.copy()
                        bp_mask[bp_mask != body_part] = 0. # only body part mask is considered
                        bp_mask = get_minimal_bbox(bp_mask, body_part)
                        bp_maskIm = cv2.resize(bp_mask, (int(x2-x1),int(y2-y1)), interpolation=cv2.INTER_NEAREST)

                        y, x = np.where(bp_maskIm > 0.)
                        bp_x1, bp_y1, bp_x2, bp_y2 =  int(np.min(x)+x1), int(np.min(y)+y1), int(np.max(x)+x1), int(np.max(y)+y1)
                        box = [bp_x1, bp_y1, bp_x2-bp_x1, bp_y2-bp_y1]

                        if box[2] <= 0 or box[3] <= 0:
                            del bp_mask
                            continue

                        assert(box[0] < width and box[0] >= 0)
                        assert(box[1] < height and box[1] >= 0)
                        assert(box[0] + box[2] < width)
                        assert(box[1] + box[3] < height)
                        assert(box[2] > 0 and box[3] > 0) 

                        # create new annotation
                        new_id = int(str(ann["id"]) + str(00) + str(body_part))
                        if new_id in ann_ids or new_id in n_ids:
                            raise ValueError("id already exists.")
                        n_ids.append(new_id)
                        
                        if body_part == 1:
                            body_part = 2
                        elif body_part == 2:
                            body_part = 3
                        elif body_part == 6 or body_part == 7:
                            body_part = 5
                        elif body_part == 8 or body_part == 9:
                            body_part = 6
                        elif body_part == 10 or body_part == 11:
                            body_part = 7
                        elif body_part == 12 or body_part == 13:
                            body_part = 8
                        elif body_part == 14:
                            body_part = 9
    
                        body_part -= 1

                        new_ann = {"area": box[2]*box[3], "iscrowd": 0, "image_id": im_id, "bbox": box, "category_id": body_part, 
                                   "id": new_id}
                        new_anns["annotations"].append(new_ann)

                        del bp_mask
                else:
                    raise ValueError()

    with open(os.path.join(args.new_anns_dir, args.new_anns_file), "w") as out:
        json.dump(new_anns, out, indent=4)
    print("Annotations created.")
    
def use_densePose_dataset_multilabel(args, new_anns):

    dp_coco = COCO(args.old_anns)
    im_ids = dp_coco.getImgIds()

    # only keep images where all persons have GT body-part segmentation masks
    percentage_bp_ann = []
    for id in im_ids:
        img = dp_coco.loadImgs(id)[0]
        ann_ids = dp_coco.getAnnIds(imgIds=img['id'])
        annotations = dp_coco.loadAnns(ann_ids)
        dp_ann = 0
        bbox_ann = 0
        for ann in annotations:
            if ann["category_id"] != 1:
                continue
            bbox_ann += 1
            if('dp_masks' in ann.keys()):
                dp_ann += 1

        if bbox_ann > 0:
            percentage_bp_ann.append(dp_ann/bbox_ann)
        
    np_percentage_bp_ann = np.array(percentage_bp_ann)
    keep = np_percentage_bp_ann == 1.
    im_ids = np.array(im_ids)[keep].tolist()
    
    print(f'Number of samples: {len(im_ids)}')

    # Load annotations
    for im_id in im_ids:

        im = dp_coco.loadImgs(im_id)[0]
        new_anns["images"].append(im)

        ann_ids = dp_coco.getAnnIds(imgIds=im['id'])
        anns = dp_coco.loadAnns(ann_ids)

        n_ids = []

        for ann in anns: 
            
            if 'dp_masks' in ann.keys():

                width, height = im["width"], im["height"]
                bbr =  np.array(ann['bbox']).astype(int)
                x1,y1,x2,y2 = bbr[0],bbr[1],bbr[0]+bbr[2],bbr[1]+bbr[3]

                Mask = GetDensePoseMask(ann['dp_masks'])

                new_ann = {"area": ann["area"], "iscrowd": ann['iscrowd'], "image_id": im_id, "bbox": ann['bbox'], "category_id": 1, "id": ann["id"]}
                new_anns["annotations"].append(new_ann)

                for body_part in map_DensePose.keys():

                    if body_part not in Mask:
                        continue

                    bp_mask = Mask.copy()
                    bp_mask[bp_mask != body_part] = 0. # only body part mask is considered
                    bp_mask = get_minimal_bbox(bp_mask, body_part)

                    bp_maskIm = cv2.resize(bp_mask, (int(x2-x1),int(y2-y1)), interpolation=cv2.INTER_NEAREST)

                    y, x = np.where(bp_maskIm > 0.)
                    bp_x1, bp_y1, bp_x2, bp_y2 =  int(np.min(x)+x1), int(np.min(y)+y1), int(np.max(x)+x1), int(np.max(y)+y1)
                    box = [bp_x1, bp_y1, bp_x2-bp_x1, bp_y2-bp_y1]

                    if box[2] <= 0 or box[3] <= 0:
                        del bp_mask
                        continue

                    assert(box[0] < width and box[0] >= 0)
                    assert(box[1] < height and box[1] >= 0)
                    assert(box[0] + box[2] < width)
                    assert(box[1] + box[3] < height)
                    assert(box[2] > 0 and box[3] > 0) 

                    # create new annotation
                    new_id = int(str(ann["id"]) + str(00) + str(body_part))
                    if new_id in ann_ids or new_id in n_ids:
                        raise ValueError("id already exists.")
                    n_ids.append(new_id)

                    if body_part == 1:
                        body_part = 2
                    elif body_part == 2:
                        body_part = 3
                    elif body_part == 6 or body_part == 7:
                        body_part = 5
                    elif body_part == 8 or body_part == 9:
                        body_part = 6
                    elif body_part == 10 or body_part == 11:
                        body_part = 7
                    elif body_part == 12 or body_part == 13:
                        body_part = 8
                    elif body_part == 14:
                        body_part = 9

                    new_ann = {"area": box[2]*box[3], "iscrowd": 0, "image_id": im_id, "bbox": box, "category_id": body_part, "id": new_id}
                    new_anns["annotations"].append(new_ann)

                    del bp_mask

    with open(os.path.join(args.new_anns_dir, args.new_anns_file), "w") as out:
        json.dump(new_anns, out, indent=4)
    print("Annotations created.")
    
def use_full_coco2014_dataset(args, new_anns):

    coco = COCO(args.old_anns)
    im_ids = coco.getImgIds()
    
    # Load annotations
    for im_id in im_ids:
        
        person_present=False

        im = coco.loadImgs(im_id)[0]

        ann_ids = coco.getAnnIds(imgIds=im['id'])
        anns = coco.loadAnns(ann_ids)
        
        tmp_anns = []
        too_small_bbox = False
        for ann in anns: 
            if ann['category_id'] == 1:
                person_present = True
                new_ann = {"area": ann["area"], "iscrowd": ann['iscrowd'], "image_id": im_id, "bbox": ann['bbox'], "category_id": 1, "id": ann["id"]}
                if ann['area'] < MIN_AREA:
                    too_small_bbox = True
                    break
                tmp_anns.append(new_ann)
                
        if too_small_bbox:
            continue
        new_anns["annotations"].extend(tmp_anns)   
          
        if person_present:
            new_anns["images"].append(im)
    
    print(f'Dataset size: {len(new_anns["images"])}')
    with open(os.path.join(args.new_anns_dir, args.new_anns_file), "w") as out:
        json.dump(new_anns, out, indent=4)
    print("Annotations created.")

if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Creating Annotations')
    args.add_argument('--anns_mode', type=str,
                      help='Annotation mode, options: person, parts, multilabel, all_persons_person, all_persons_parts, all_persons_multilabel, voc2010_person, voc2010_parts, voc2010_multilabel')
    args.add_argument('--old_anns', default="", type=str,
                      help='Path to DensePose annotation file (default: None)')
    args.add_argument('--new_anns_dir', default="/storage/project_data/robust_object_detection_paper/annotations/concatenated_dataset/coco_multilabel", type=str,
                      help='Path to new annotations directory (default: None)')
    args.add_argument('--new_anns_file', default="val.json", type=str,
                      help='Filename of new annotations (default: None)')

    args = args.parse_args()
    
    if args.anns_mode not in ["person", "parts", "multilabel", "all_persons_person", "all_persons_parts", "all_persons_multilabel", 
                              "voc2010_person", "voc2010_parts", "voc2010_multilabel"]:
        raise NotImplementedError("Unsupported annotation mode.")

    if not os.path.exists(args.new_anns_dir):
        os.makedirs(args.new_anns_dir)

    new_anns = None

    # DensePose annotations
    if args.anns_mode == "person":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_person.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_densePose_dataset(args, new_anns)
    elif args.anns_mode == "parts":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_reduced_DensePose.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_densePose_dataset(args, new_anns)
    elif args.anns_mode == "multilabel":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_multilabel.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_densePose_dataset_multilabel(args, new_anns)
    # COCO2014 annotations
    elif args.anns_mode == "all_persons_person":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_person.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_full_coco2014_dataset(args, new_anns)
    elif args.anns_mode == "all_persons_parts":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_reduced_DensePose.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_full_coco2014_dataset(args, new_anns)
    elif args.anns_mode == "all_persons_multilabel":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_multilabel.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_full_coco2014_dataset(args, new_anns)
    # PascalVOC 2010 annotations
    elif args.anns_mode == "voc2010_multilabel":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_multilabel.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_annotations_voc2010(args, new_anns)
    elif args.anns_mode == "voc2010_person":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_person.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_annotations_voc2010(args, new_anns)
    elif args.anns_mode == "voc2010_parts":
        new_anns = {"categories": [], "images": [], "annotations": []}
        for k, v in map_reduced_DensePose.items():
            cat = {"supercategory": v, "id": k, "name": v}
            new_anns["categories"].append(cat)
        use_annotations_voc2010(args, new_anns)
    else:
        raise ValueError("Unsupported dataset.")
