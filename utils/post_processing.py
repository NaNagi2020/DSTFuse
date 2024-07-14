import numpy as np

target = [1]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    union_area = area_box1 + area_box2 - intersection_area

    iou = intersection_area / union_area

    return iou

def threshold_filtering(preds, threshold):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    if boxes.device == 'cpu':
        boxes = boxes.numpy()
        scores = scores.numpy()
        labels = labels.numpy()
    else:
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()

    indices_to_keep = np.logical_and(scores >= threshold, np.isin(labels, target))
    boxes = boxes[indices_to_keep]
    scores = scores[indices_to_keep]
    labels = labels[indices_to_keep]
    assert len(scores) == len(boxes)
    return boxes, scores
        

def nms(boxes, threshold):
    
    res = np.empty((0, 4))
    while(len(boxes) > 0):
        cur_box = boxes[0]
#         if np.all(cur_box < 1):
#             boxes = np.delete(boxes, 0, axis=0)
#             continue
        res = np.vstack([res, cur_box])
        boxes = np.delete(boxes, 0, axis=0)
        del_cond = [calculate_iou(cur_box, box) > threshold for box in boxes]
#         ious = [calculate_iou(cur_box, box) for box in boxes]
        boxes = np.delete(boxes, del_cond, axis=0)
#     print(res)
    return res

def enlarge_boxes(boxes):
    im_size = 640
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]

    new_width = width * 1.5
    new_height = height * 1.5
    new_boxes = np.zeros_like(boxes)
    new_boxes[:, 0] = center_x - new_width / 2
    new_boxes[:, 1] = center_y - new_height / 2
    new_boxes[:, 2] = center_x + new_width / 2
    new_boxes[:, 3] = center_y + new_height / 2

    new_boxes[:, 0] = np.maximum(0, new_boxes[:, 0])
    new_boxes[:, 1] = np.maximum(0, new_boxes[:, 1])
    new_boxes[:, 2] = np.minimum(new_boxes[:, 2], im_size)  # 假设图像尺寸为 256x256
    new_boxes[:, 3] = np.minimum(new_boxes[:, 3], im_size)  # 假设图像尺寸为 256x256

    return new_boxes

def post_process(prediction_vis, prediction_ir):
    boxes = np.empty((0, 4))
    for pred in prediction_vis:
        box, score = threshold_filtering(pred, 0.9)
        box = nms(box, 0.5)
        boxes = np.vstack([boxes, box])
        
    for pred in prediction_ir:
        box, score = threshold_filtering(pred, 0.9)
        box = nms(box, 0.5)
        boxes = np.vstack([boxes, box])
    boxes = enlarge_boxes(nms(boxes, 0.5))
    return boxes


if __name__ == "__main__":
    import os
    import sys
    import time
    import datetime

    import torch
    import torch.nn as nn
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    from skimage.color import rgb2gray
    from skimage.io import imread
    import cv2

    from visualize import visualize


    # det_model_vis1 = torch.load('model/det_model_vis.pth')
    # det_model_ir = torch.load('model/det_model_ir.pth')
    num_classes = 91
    names = {'0': 'background', '1': 'person', '2': 'bicycle', '3': 'car', '4': 'motorcycle', '5': 'airplane', '6': 'bus', '7': 'train', '8': 'truck', '9': 'boat', '10': 'traffic light', '11': 'fire hydrant', '13': 'stop sign', '14': 'parking meter', '15': 'bench', '16': 'bird', '17': 'cat', '18': 'dog', '19': 'horse', '20': 'sheep', '21': 'cow', '22': 'elephant', '23': 'bear', '24': 'zebra', '25': 'giraffe', '27': 'backpack', '28': 'umbrella', '31': 'handbag', '32': 'tie', '33': 'suitcase', '34': 'frisbee', '35': 'skis', '36': 'snowboard', '37': 'sports ball', '38': 'kite', '39': 'baseball bat', '40': 'baseball glove', '41': 'skateboard', '42': 'surfboard', '43': 'tennis racket', '44': 'bottle', '46': 'wine glass', '47': 'cup', '48': 'fork', '49': 'knife', '50': 'spoon', '51': 'bowl', '52': 'banana', '53': 'apple', '54': 'sandwich', '55': 'orange', '56': 'broccoli', '57': 'carrot', '58': 'hot dog', '59': 'pizza', '60': 'donut', '61': 'cake', '62': 'chair', '63': 'couch', '64': 'potted plant', '65': 'bed', '67': 'dining table', '70': 'toilet', '72': 'tv', '73': 'laptop', '74': 'mouse', '75': 'remote', '76': 'keyboard', '77': 'cell phone', '78': 'microwave', '79': 'oven', '80': 'toaster', '81': 'sink', '82': 'refrigerator', '84': 'book', '85': 'clock', '86': 'vase', '87': 'scissors', '88': 'teddybear', '89': 'hair drier', '90': 'toothbrush'}
    # names = {'0': 'background', '1': 'person'}
    # num_classes = 2  # 背景类别 + person 类别

    det_model_vis = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=True)
    det_model_ir = torchvision.models.detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes, pretrained=True)


    det_model_vis.eval()
    input = []
    img_tensor = torch.from_numpy(resize(rgb2gray(imread("data/190060.jpg")), (640, 640))).float().unsqueeze(0)
    input.append(img_tensor)
    out = det_model_vis(input)
    print(out)
    visualize(out, "data/190060.jpg")
    for pred in out:
        boxes, scores = threshold_filtering(pred, 0.7)
    #     print(boxes, scores)
        nms(boxes, 0.5)
        visualize(boxes, "data/190060.jpg")