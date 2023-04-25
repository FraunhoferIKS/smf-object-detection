import argparse
import os
import pathlib

from src.tools.evaluation import evaluate, compute_metrics
from src.tools.visualization import visualize_distribution
from src.tools.utils import map_person, map_multilabel, map_reduced_DensePose

def performance_characteristics_tradeoff(args, mapping: dict):
    """Visualizes the precision, recall, and f1-score for the confidence score thresholds with best precision-recall trade-off"""
    
    l_f1_score, l_score, l_precision, l_recall = [], [], [], []
    
    scores, precision, recall, f1_score = compute_metrics(path_to_anns=args.path_anns, path_to_dets=args.path_dets)
    
    for i in mapping.keys():

        # get optimal precision recall threshold (both weighted the same)
        prec, rec, f1, sc = precision[:, i-1].tolist(), recall.tolist(), f1_score[:, i-1].tolist(), scores[:, i-1].tolist()

        max_point = max(f1)
        id = f1.index(max_point)

        l_f1_score.append(max_point)
        l_score.append(sc[id])
        l_precision.append(prec[id])
        l_recall.append(rec[id])

    visualize_distribution(y=l_f1_score, mapping=mapping, save_dir=args.save_path, filename='distribution_f1_score_tradeoff.jpg', ylabel='F1-Score')
    visualize_distribution(y=l_score, mapping=mapping, save_dir=args.save_path, filename='distribution_conf_thresh_tradeoff.jpg', ylabel='Confidence Threshold')
    # save values in .txt file
    with open(os.path.join(args.save_path, f'distribution_conf_thresh_tradeoff.txt'), 'w') as f:
        f.write(f'Mapping: {mapping.values()} \n')
        for s in l_score:
            f.write(f'{s} ')
    visualize_distribution(y=l_precision, mapping=mapping, save_dir=args.save_path, filename='distribution_precision_tradeoff.jpg', ylabel='Precision')
    visualize_distribution(y=l_recall, mapping=mapping, save_dir=args.save_path, filename='distribution_recall_tradeoff.jpg', ylabel='Recall')

def compute_mAP_per_class(args, mapping: dict):

    l_mAP = []
    l_mAP_05 = []
    l_recall_1 = []
    l_recall_10 = []

    for i in mapping.keys():

        print('-------------------------------------------------------------')
        print(f'Performance for class {mapping[i]}:')
        print('-------------------------------------------------------------')

        stats = evaluate(path_to_anns=args.path_anns, path_to_dets=args.path_dets, cats=[i])
        l_mAP.append(stats[0])
        l_mAP_05.append(stats[1])
        l_recall_1.append(stats[6])
        l_recall_10.append(stats[7])

    visualize_distribution(y=l_mAP, mapping=mapping, save_dir=args.save_path, filename='distribution_per_class_mAP.jpg', ylabel='mAP')
    
    with open(os.path.join(args.save_path, f'stats.txt'), 'w') as f:
        
        f.write(f'################################################################################################ \n')
        f.write(f'Classes: \n')
        f.write(f'################################################################################################ \n')
        f.write(f'{list(mapping.keys())} \n')
        f.write(f'{[v for v in mapping.values()]} \n')
        f.write(f'################################################################################################ \n')
        f.write(f'mAP per class: \n')
        f.write(f'################################################################################################ \n')
        f.write(f'{l_mAP} \n')
        f.write(f'################################################################################################ \n')
        f.write(f'mAP@0.5 per class: \n')
        f.write(f'################################################################################################ \n')
        f.write(f'{l_mAP_05} \n')
        f.write(f'################################################################################################ \n')
        f.write(f'AR with maxDet=1 per class: \n')
        f.write(f'################################################################################################ \n')
        f.write(f'{l_recall_1} \n')
        f.write(f'################################################################################################ \n')
        f.write(f'AR with maxDet=10 per class: \n')
        f.write(f'################################################################################################ \n')
        f.write(f'{l_recall_10} \n')
        
if __name__ == "__main__":

    args = argparse.ArgumentParser(description='Model Characteristics')

    args.add_argument('--path_anns', default=None, type=str)
    
    args.add_argument('--path_dets', default=None, type=str)
    
    args.add_argument('--save_path', default=None, type=str)

    args.add_argument('--mode', default=None, type=str, help="Options: person, parts, multilabel")

    args = args.parse_args()
    
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)

    mapping=None
    num_classes=None
    if args.mode == "person":
        mapping=map_person
        num_classes=1
    elif args.mode == "parts":
        mapping=map_reduced_DensePose
        num_classes=8
    elif args.mode == "multilabel":
        num_classes=9
        mapping=map_multilabel
    else:
        raise ValueError("Unsupported mode, only 'person', 'parts', and 'multilabel' are possible.")

    compute_mAP_per_class(args, mapping=mapping)
    performance_characteristics_tradeoff(args, mapping=mapping)
    