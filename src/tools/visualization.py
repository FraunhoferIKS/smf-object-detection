import matplotlib.pyplot as plt
import os
from typing import List, Dict, Any
import numpy as np
import bbox_visualizer as bbv
import seaborn as sns
from pathlib import Path

def visualize_distribution(y: list, mapping: dict, save_dir: str, filename: str, ylabel: str):

    x = []
    for i in mapping.keys():
        x.append(mapping[i])

    fig = plt.figure(figsize=(25,5))
    ax = fig.add_subplot(1,1,1)

    r_y = [np.round(item, decimals=2) for item in y]

    sns.barplot(x=x, y=y, palette="Blues_d", ax=ax)
    ax.bar_label(ax.containers[0], labels=r_y, padding=3)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Classes')
    plt.savefig(os.path.join(save_dir, filename))

def draw_bounding_boxes(img: np.array, boxes: np.array, mapping: Dict=None, labels: List[int]=None, scores: List[float]=None, color: Any = (255, 255, 255)):

    vis_boxes = []
    boxes = boxes.astype(int)
    if len(boxes.shape) == 2:
        for i in range(boxes.shape[0]):
            vis_boxes.append(boxes[i, :].tolist())
        
    if len(vis_boxes):
        img = bbv.draw_multiple_rectangles(img, vis_boxes, bbox_color=color)

        class_labels = None

        if labels is not None and mapping is not None:
            class_labels = [mapping[i] for i in labels]

        if scores is not None:
            class_labels = [f'{item}: {np.round(scores[i]*100, decimals=1)}%' for i, item in enumerate(class_labels)]

        if class_labels is not None:
            img = bbv.add_multiple_labels(img, class_labels, vis_boxes, top=False, text_bg_color=color)

    return img