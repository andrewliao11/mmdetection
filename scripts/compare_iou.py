import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import ipdb


def get_annotations_from_image_id(image_id, labels):
    annotations = [anno for anno in labels["annotations"] if anno["image_id"] == image_id]
    return annotations

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def xywh_to_xyxy(xywh):
    x_top_left, y_top_left, width, hieght = xywh
    x_bottom_right = x_top_left + width
    y_bottom_right = y_top_left + hieght
    return [x_top_left, y_top_left, x_bottom_right, y_bottom_right]


def main():

    parser = argparse.ArgumentParser(description="Prepare datasets")
    parser.add_argument("--labels_1_path", type=str)
    parser.add_argument("--labels_2_path", type=str)
    args = parser.parse_args()

    # assuming that two datasets have the same amount of images, same category space
    labels_1 = json.load(open(args.labels_1_path))
    labels_2 = json.load(open(args.labels_2_path))


    image_ids = [i["id"] for i in labels_1["images"]]


    all_boxes_iou = []
    for image_id in tqdm(image_ids):
        annotations_1 = get_annotations_from_image_id(image_id, labels_1)
        annotations_2 = get_annotations_from_image_id(image_id, labels_2)
        if len(annotations_1) == 0 and len(annotations_2) == 0:
            continue

        # class agnostic (assuming that there is no class mismatch, only box mismatch)
        # iou_matrix: (|annotations_1|, |annotations_2|)
        iou_matrix = np.array([[bb_intersection_over_union(xywh_to_xyxy(a1["bbox"]), xywh_to_xyxy(a2["bbox"])) for a2 in annotations_2] for a1 in annotations_1])
        #TODO: Does not work when iou_matrix is not squared
        if len(iou_matrix.shape) == 2:
            all_boxes_iou.extend(iou_matrix.diagonal().tolist())
        else:
            all_boxes_iou.extend([iou_matrix[0]])


    miou = np.array(all_boxes_iou).mean()
    print(f"Mean IoU: {miou}")
                

if __name__ == "__main__":
    main()