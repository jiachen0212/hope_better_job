# 单目求体积: https://zhuanlan.zhihu.com/p/476303132
# coding=utf-8
import numpy as np  
import cv2 

KNOWN_DISTANCE = 32  
KNOWN_WIDTH = 11.69  # 物体的实际大小信息 
IMAGE_PATHS = ["Picture1.jpg", "Picture2.jpg", "Picture3.jpg"] # 将用到的图片放到了一个列表中

def find_marker(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  
    edged_img = cv2.Canny(gray_img, 35, 125)  
    # 获取纸张的轮廓数据
    img, countours, hierarchy = cv2.findContours(edged_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(countours, key=cv2.contourArea)  
    rect = cv2.minAreaRect(c)  # 检测到了目标 

    return rect

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

# 相机内参: 焦距:
def calculate_focalDistance(img_path):
    first_image = cv2.imread(img_path) 
    marker = find_marker(first_image) # 获取物体的, 中心点, h, w,旋转角 
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH  

    return focalLength

def calculate_Distance(image_path, focalLength_value):
    image = cv2.imread(image_path)
    marker = find_marker(image) # 获取矩形的中心点坐标，长度，宽度和旋转角度， marke[1][0]代表宽度
    distance_inches = distance_to_camera(KNOWN_WIDTH, focalLength_value, marker[1][0])


if __name__ == "__main__":

    img_path = "Picture1.jpg"
    # 算相机焦距
    focalLength = calculate_focalDistance(img_path)
    # 算深度距离
    calculate_Distance(image_path, focalLength)