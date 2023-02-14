# coing=utf-8
import cv2
import numpy as np
# 双线性插值算法实现 

def bilinear_fun(img, out_dim):
    src_h,src_w = img.shape[:2]
    dst_h,dst_w = out_dim[1],out_dim[0]
    # if src_h == dst_h and src_w == dst_w:
    #     return img.copy()
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y = float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range (dst_w):
                # 找到dst坐标的对应src坐标
                src_x = (dst_x+0.5)*scale_x -0.5
                src_y = (dst_y+0.5)*scale_y -0.5
                # np.floor()向下取整 
                src_xl = int(np.floor(src_x))
                src_xr = min(src_xl + 1, src_w -1)  # 防止过边界 
                src_yl = int(np.floor(src_y))
                src_yr = min(src_yl + 1,src_h -1)
               # xy两个维度上的4个weights
                temp0 = (src_xr - src_x) * img[src_yl,src_xl,i] + (src_x - src_xl) * img[src_yl,src_xr,i]
                temp1 = (src_xr - src_x) * img[src_yr,src_xl,i] + (src_x - src_xl) * img[src_yr,src_xr,i]
                dst_img[dst_y,dst_x,i] = int((src_yr - src_y) * temp0 + (src_y - src_yl) * temp1)
    return dst_img

if __name__ == '__main__':
    img = cv2.imread('/Users/chenjia/Downloads/Smartmore/2022/daydayup/图像马赛克_方块_毛边/lena.png')
    print(img.shape)
    dst_img = bilinear_fun(img, [300,300])
    cv2.imwrite('./bilinear_lena.jpg', dst_img)
