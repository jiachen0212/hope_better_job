# coding=utf-8
# 去水印算法
from os.path import splitext
# 计算笛卡尔积: 两个集合的所有组合
from itertools import product
from PIL import Image
import cv2 
import random
import numpy as np 



def qu_shuiyin(fn, pixel_thres=580):
	im = Image.open(fn)
	width, height = im.size
	for pos in product(range(width), range(height)):
	    # 这个pos其实就是所有图像点坐标
	    # 580是经验值, RGB值之和>580则认为此处是水印. 
	    if sum(im.getpixel(pos)[:3]) > pixel_thres:
	        im.putpixel(pos, (250,250,250))  # 重新pixel赋值 
	im.save('_无水印'.join((splitext(fn))))

# fn = './sy.png'
# qu_shuiyin(fn)



# coding=utf-8
print('交换a和b两个数, 不借助第三个变量')
a, b = 1,2
print("a: {}, b: {}".format(a, b))
a, b = (a+b-a), (a+a)/2
print("a: {}, b: {}".format(a, b))



print('\n')
print('lambda生成奇数数组')
def how_long_jssz(len_):
	jssz = []
	lam = lambda x: 2*x+1
	for _ in range(len_):
		jssz.append(lam(_))

	return jssz

len_ = 16
print('长度为{}的lambda奇数数组: {}'.format(len_, how_long_jssz(len_)))


print('\n')
print('精度保留5位开方')
def sqart_(x):
	if x == 0:
		return x
	X, res = x, x
	while True:
		temp = (res + X/res) / 2
		if abs(res - temp) <= 1e-7:
			break
		res = temp
	return round(res, 5)
print('sqart_(5): {}'.format(sqart_(5)))


print('\n')
print('马赛克算法: 方块mask, 毛边mask')
def image_mask(h_range, w_range, img_name, winds=8):
	lena = cv2.imread(img_name)
	for i in range(h_range[0], h_range[1]):
		for j in range(w_range[0], w_range[1]):
			# 用8x8窗口扫描,取窗口左上角value带图这8x8内的所有像素值.
			# 可起到方块马赛克作用
			if i%winds == j%winds == 0:
				mask_value = lena[i][j]
				for a_ in range(winds):
					for b_ in range(winds):
						lena[i+a_][j+b_] = mask_value
	cv2.imwrite('./mask_lena.jpg', lena)

h_range = [90, 140]
w_range = [80, 180]
img_name = '/Users/chenjia/Desktop/lena.png'
# image_mask(h_range, w_range, img_name)

def image_mask_maoboli(img_name, winds=8):
	# 让像素随机被周围像素替换, 
	lena = cv2.imread(img_name)
	# 存放mask结果图像
	dst = np.zeros(lena.shape, np.uint8)

	h, w = lena.shape[:2]
	for i in range(h):
		for j in range(w):
			# 8x8窗口内, 随机找一个像素取value,来代替[i,j]处的像素值
			off_h, off_w = random.randint(0, winds), random.randint(0, winds)
			off_h += i 
			off_w += j 
			# 图像边界越界处理:
			off_h = min(h-1, off_h)
			off_w = min(w-1, off_w)
			dst[i][j] = lena[off_h][off_w]
	cv2.imwrite('./maobian_mask_lena.jpg', dst)
# image_mask_maoboli(img_name)


print('\n')
print('k-means算法: ')
def randCent(dataSet, k):
    m,n = dataSet.shape
    centroids = np.zeros((k,n))
    for i in range(k):
        index = int(np.random.uniform(0,m))  
        centroids[i,:] = dataSet[index,:]
    return centroids
def distEclud(x, y):
	return np.sqrt(np.sum((x-y)**2))
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]   
    # 创建矩阵, 存储: 哪一簇, 到簇中心的dis
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True

    # 初始化簇中心 
    centroids = randCent(dataSet, k)
    while clusterChange: 
        clusterChange = False
        for i in range(m):
            minDist = 100000.0
            minIndex = -1
            # 找出最近的中心
            for j in range(k):
                distance = distEclud(centroids[j,:],dataSet[i,:])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 更新所属簇
            if clusterAssment[i,0] != minIndex:
                clusterChange = True
                clusterAssment[i,:] = minIndex, minDist**2
        # 更新中心
        for j in range(k):
        	# .A操作是把matrix转为ndarray
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j,:] = np.mean(pointsInCluster,axis=0)  # 求均值给到中心

    return centroids, clusterAssment


print('\n')
print('均值滤波, 高斯滤波: ')
def slow_MeanFilter(im,r):
    H,W = im.shape
    res = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            s,n=0,0
            for k in range(i-r//2,i+r-r//2):
                for m in range(j-r//2,j+r-r//2):
                    if k<0 or k>=H or m<0 or m>=W:
                        continue
                    else:
                        s += im[k,m]
                        n += 1
            res[i,j] = s/n
    return res
    
def MeanFilter(im, r):
    '''
    im灰度图, r为滤波半径

    '''

    H,W = im.shape 
    res = np.zeros((H,W))
    
    # 行维度上, 由上至下逐渐加法, 做行上的积分
    integralImagex = np.zeros((H+1,W))
    for i in range(H):    
        integralImagex[i+1,:] = integralImagex[i,:]+im[i,:]
    # 以r为单位, 对应做减法, 起到下面多一行, 上面就减一行的效果 
    # 得到的mid, 就是r行上, 各行的积分值(完成了行上加法)
    mid = integralImagex[r:]-integralImagex[:-r]
    # /r 就是在行维度上做mean处理   
    mid = mid / r  

    # 行上做左右padding
    padding = r - 1 
    leftPadding = (r-1)//2 
    rightPadding = padding - leftPadding

    # 基本后第i行值padding
    left = integralImagex[r-leftPadding:r]
    # 原im的值padding
    right = integralImagex[-1:] - integralImagex[-r+1:-r+1+rightPadding]

    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(-1,1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(-1,1)
    left /= leftNorm
    right /= rightNorm
    im1 = np.concatenate((left,mid,right))

    # 相同方式处理列
    integralImagey = np.zeros((H,W+1))
    res = np.zeros((H,W))
    for i in range(W):    
        integralImagey[:,i+1] = integralImagey[:,i]+im1[:,i]
    mid = integralImagey[:,r:]-integralImagey[:,:-r] 
    mid = mid / r 
    left = integralImagey[:,r-leftPadding:r]
    right = integralImagey[:,-1:] - integralImagey[:,-r+1:-r+1+rightPadding]
    leftNorm = np.array(range(r-leftPadding,r,1)).reshape(1,-1)
    rightNorm = np.array(range(r-1,r-rightPadding-1,-1)).reshape(1,-1)
    left /= leftNorm
    right /= rightNorm
    im2 = np.concatenate((left,mid,right),axis=1)
    
    return im2

print('\n')

def add_gauss(im):
  H, W = im.shape[:2]
  mean = 0
  sigma = 5
  gauss = np.random.normal(mean,sigma,(H,W))
  noisy_img = im + gauss
  noisy_img = np.clip(noisy_img,a_min=0,a_max=255)

  return noisy_img

def gauss_kernel(n, sigma=1):
  gauss_wind = [[0 for i in range(n)] for j in range(n)]
  for i in range(n):
    for j in range(n):
      gauss_wind[i][j] = 1/(2*np.pi*sigma**2)*np.exp(-(i**2+j**2)/(2*sigma**2)) 

  return gauss_wind

def run_gauss():
    im = cv2.imread('./lena.png', 0)
    im = add_gauss(im)
    cv2.imwrite('./gauss_lena.jpg', im)
    r = 5
    H, W = im.shape[:2]
    res = np.zeros((H,W))
    for i in range(H-r):
      for j in range(W-r):
        im_ = im[i: i+r, j:j+r]
        res[i: i+r, j:j+r] = np.matmul(im_, gauss_kernel(r, sigma=0.5))
    cv2.imwrite('./gauss_filter_lena.jpg', res)


print('\n')
print('画灰度图像的直方图:')
import matplotlib.pyplot as plt
def histogram(im): 
    ret = [0]*256
    x, y = im.shape[:2]
    for i in range(x):  
        for j in range(y):
            ret[im[i][j]] += 1
    plt.bar([i for i in range(256)], ret)
    plt.show()
# im = cv2.imread('/Users/chenjia/Downloads/ycy_better_work/NMS/IMG_6831.JPG', 0)
# histogram(im)


print('\n')








