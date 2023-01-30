import cv2
import matplotlib.pyplot as plt

def histogram(im): 
    ret = [0]*256
    x, y = im.shape[:2]
    for i in range(x):  
        for j in range(y):
            ret[im[i][j]] += 1
    plt.bar([i for i in range(256)], ret)
    plt.show()

im = cv2.imread('/Users/chenjia/Downloads/ycy_better_work/NMS/IMG_6831.JPG', 0)
histogram(im)