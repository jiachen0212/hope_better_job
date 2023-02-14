# coing=utf-8
import cv2
import numpy as np

'''
iou在分割中, 依旧是交并比 : tp/(fp+fn+tp)   预测_真实
混淆矩阵计算: x:预测类别, y:真实类别

https://www.jianshu.com/p/42939bf83b8a?ivk_sa=1024320u

'''

def per_class_iou(hist):
    # hist 混淆矩阵(n, n)
    np.seterr(divide="ignore", invalid="ignore")
    # 交集: np.diag取hist的对角线: tp 
    # 并集: hist.sum(1)和hist.sum(0)分别按两个维度相加, 而对角线元素加了两次,so减一次 
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iou
