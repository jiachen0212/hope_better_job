# coding=utf-8
import os 
import cv2
import numpy as np 
from skimage import exposure

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(image, table)


dir_ = '/Users/chenjia/Desktop/1'
ims = [os.path.join(dir_, a) for a in os.listdir(dir_)]
res_ims = []
for name in ims:
    im1 = cv2.imread(name)
    # gam1= exposure.adjust_gamma(im1, 0.4)
    # (b, g, r) = cv2.split(im1)
    # bH = cv2.equalizeHist(b)
    # gH = cv2.equalizeHist(g)
    # rH = cv2.equalizeHist(r)
    # gam1 = cv2.merge((bH, gH, rH))
    gam1 = adjust_gamma(im1, gamma=5)
    # cv2.imwrite('./1.jpg', gam1)
    res_ims.append(gam1)
rep_ims = []
for i in range(17):
    rep_ims.extend(res_ims)
# 写入视频
video = cv2.VideoWriter("./chenj.mp4", cv2.VideoWriter_fourcc('m','p','4','v'), 1, (320, 240))
for img in rep_ims:
    # img = cv2.cvtColor(img.astype(np.uint8).transpose(), cv2.COLOR_GRAY2BGR)
    video.write(img)
video.release()

