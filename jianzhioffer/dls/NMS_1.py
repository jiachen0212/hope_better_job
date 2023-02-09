# coding=utf-8
# https://github.com/hiJulie/NMS/blob/master/non_max_suppress.py
import numpy as np

def non_max_suppress(predicts_dict, thr):
    for object_name, bbox in predicts_dict.items():  # class and it's bbox: key, value
        bbox_array = np.array(bbox, dtype=np.float)
        x1 = bbox_array[:, 0]
        y1 = bbox_array[:, 1]
        x2 = bbox_array[:, 2]
        y2 = bbox_array[:, 3]
        scores = bbox_array[:, 4]
        order = scores.argsort()[::-1]   # 降序排序并返回index
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        keep = []

        while order.size > 0:
            i = order[0]     # i 是当前box index 
            keep.append(i)  
            # np.maximum: 逐元素比较两array中的大值
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
            iou = inter / (areas[i] + areas[order[1:]] - inter)   # iou=交集面积/(俩面积-交集面积)
            inds = np.where(iou <= thr)[0]  # if iou <= thr save the box.
            order = order[inds + 1] # 注意这个+1
        bbox = bbox_array[keep]
        predicts_dict[object_name] = bbox.tolist()

    return predicts_dict

