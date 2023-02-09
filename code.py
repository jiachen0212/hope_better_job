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
print('精度保留5位开方/开根号')
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


# 2023.01.31shuati!
# 面试code准备  蘑菇车联
# 最大正方形 leetcode221
def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0
    # dp[i][j]: [i,j]位置的左上角所有元素中, 最大正方形面积(也即1的个数)
    m, n = len(matrix), len(matrix[0])
    dp = [[0]*(n+1) for _ in range(m+1)]
    bianchang = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if matrix[i-1][j-1] == '1':
                # [i,j]左,上,左上三个位置找最小, 然后才能加上[i-1][j-1]这个位置的'1'(+1)
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
                bianchang = max(bianchang, dp[i][j])
    return bianchang**2
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
print('最大正方形面积, 最多个1组成的正方形: ', maximalSquare(matrix))
# 有序连表合并



############################################################
# 哈希表字典, 前缀和, 滑动窗
# 和等于 k 的最长子数组长度
# dict字典哈希表, 前缀和, 滑动窗
def maxSubArrayLen(nums, k):
    # sum_start_index = dict() key:连续小组的sum, value:和为sum的起始index
    # res 连续和为k的最长长度
    # sum_ 数组逐渐遍历累积和
    sum_start_index = {0: -1}
    sum_ = 0
    res = 0
    for ind, val in enumerate(nums):
        sum_ += val
        if sum_ not in sum_start_index:
            sum_start_index[sum_] = ind 
        # dict中存在某个key-v对, 是的v+val=k 
        # 则sum_-k到sum_内的和就是k了
        if sum_ - k in sum_start_index:
            res = max(res, ind-sum_start_index[sum_-k])
    return res 
nums = [-2,-1,2,1]
k = 1
print("和为k的连续数组, 最长可以多长? :", maxSubArrayLen(nums, k))

# 连续数组最大和
def maxSubArray(nums):
    # 取决于dp[i-1]的正负: dp[i] = max(dp[i-1]+nums[i], nums[i])
    lens = len(nums)
    pre, res = 0, nums[0]
    for num in nums:
        pre = max(num, num+pre)
        res = max(pre, res)
    return res 
print('连续数组最大和: ', maxSubArray([5,4,-1,7,8]))

# 滑动窗 哈希表
# 至多 包含k个不同字符的, 最长连续子串
def lengthOfLongestSubstringTwoDistinct(s, k):
    lens = len(s)
    count = [0]*256  # 把字符转为数字, 在对应index处技术
    l, r = 0, 0
    diff_cnt = 0
    res = 0
    # 不断外扩r, 到>=3不等字符,开始内缩l(l+1)
    while r < lens:
        # 计数值0才需要diff_cnt+1: 0才代表新的字符
        if count[ord(s[r])] == 0: # init时候orl内缩到字符计数清零
            diff_cnt += 1
        count[ord(s[r])] += 1
        while diff_cnt > k:
            count[ord(s[l])] -= 1
            if count[ord(s[l])] == 0:
                diff_cnt -= 1
            l += 1
        r += 1
        res = max(res, r-l)
    return res
s, k = 'eceba', 2
print('至多, 最多含有2个不同字符的, 最长连续子串: ', lengthOfLongestSubstringTwoDistinct(s, k))

# 



# 和为k的连续子数组的, 个数
# 哈希表, 前缀和, 数组
def subarraySum(nums, k):
    # sum_start_index = dict() key:连续小组的sum, value:和为sum的次数. 
    # 考虑数组正负交替会出现这样的.
    # sum_ 数组逐渐遍历累积和
    sum_start_index = {0: 1}   # init的时候注意下, 赋值为和为0次数为1, 保证k=0情况的正确性.
    sum_ = 0
    res = 0
    for num in nums:
        sum_ += num
        # 这一行就是, 更新前缀和为sum_的次数.
        res += sum_start_index.get(sum_ - k, 0)
        sum_start_index[sum_] = sum_start_index.get(sum_, 0) + 1
    return res 
nums, k = [1,2,3], 3
print('和为k的连续子数组, 有多少对: ', subarraySum(nums, k))

# 最少交换次数来组合所有的 1
# 滑动窗: 窗口大小为1的个总数, 从左往右遍历数组, 更新每个位置处需要换几次.
# 换的次数怎么算: 1的总个数-滑动窗内1的个数
def minSwaps(data):
    lens = len(data)
    # 记录[0,i]内1的个数, 遍历后续算窗口内的1个数.
    pre_one_nums = [0]*(lens+1)
    for i in range(lens):
        pre_one_nums[i+1] = pre_one_nums[i]+data[i]

    all_ones = pre_one_nums[-1]
    # 初始化最小交换次数, 不会大于1的数个+1的
    res = all_ones+1

    # 从all_ones-1index处更新右滑([0~all_ones-1]是初始窗口长为all_ones)
    for i in range(all_ones-1, lens):
        # i+1-all_ones为窗口左边界, i+1为窗口右边界
        wind_one_nums = pre_one_nums[i+1] - pre_one_nums[i+1-all_ones]
        # 交换次数: all_ones - wind_one_nums. 也就是还差多少个1
        res = min(res, all_ones - wind_one_nums)
    return res 
data = [1,0,1,0,1,0,0,1,1,0,1]
print('最少交换次数来组合所有的 1', data, minSwaps(data))

#  0,1个数相同的子数组,的长度
# 前缀和, 哈希表
def findMaxLength(nums):
    # 存和为sum的连续子数组的,起始index
    pre_sum = {0:-1}
    ans = 0
    sum_ = 0
    for i, num in enumerate(nums):
        # 起到把0变成1的效果, 则01个数相等变为数组之和为0 
        sum_ += 1 if num == 1 else -1
        if sum_ in pre_sum:
            ans = max(ans, i-pre_sum.get(sum_))
        else:
            pre_sum[sum_] = i
    return ans 

# 和>=target的最短连续子数组
# 也是滑动窗思想, ij作为窗的左右边. j用for循环遍历完数组, 当sum超过了, i则也往右锁减少窗长度
def minSubArrayLen(nums, target):
    lens = len(nums)
    i = 0
    sum_ = 0
    # 最大正值
    res =  2**32-1
    for j in range(lens):
        sum_ += nums[j]
        # 持续往右扩窗指导sum超过target
        # i<=j是保证窗口至少有1个值没缩没了, 
        while i<=j and sum_ >= target:
            # 更新下现在的窗口长度
            res= min(res, j+1-i)
            # 开始内缩窗口左边界
            sum_ -= nums[i]  
            i += 1
    return res if res < 2**32-1 else 0  
target, nums = 7, [2,3,1,2,4,3]
print('数组中和>=target的最小连续子组长度: ', minSubArrayLen(nums, target))
    

# 旋转数组找target  先升序后降序  [是一个升序数组把前一部分搬到后面去]
def search(nums, target):
    if not nums:
        return -1
    l, r = 0, len(nums)-1
    # 数组中有重复元素的话
    # while l<r and nums[l]==nums[l+1]: 
    #         l += 1
    # while l<r and nums[r]==nums[r-1]:
    #     r -= 1
    while l<=r:
        mid = (l+r)//2
        if nums[mid] == target:
            return mid
        elif nums[mid] < nums[r]:  # mid~r有序
            if nums[mid] < target <= nums[r]:
                l = mid+1   # target在右边, 加大左边界
            else:   # target没在这段有序内, 在左, 则减小r
                r = mid - 1
        else:  # 左边有序
            if nums[l] <= target < nums[mid]:
                r = mid-1
            else:
                l = mid+1
    return -1 
nums, target = [5,7,8,10,3], 3
print('旋转数组找target的index: ', search(nums, target))

# 乘积<k的连续子数组 有多少对
def sub_bins_less_target(nums, k):
    # 还是[i,j]不定长滑窗思想
    lens = len(nums)
    i, j = 0, 0
    res = 0
    mul = 1
    while j < lens:
        mul *= nums[j]
        j += 1
        if mul < k:
            res += (j-i)
        else:
            # [i,j]内的乘积大于k了, 则内缩左边界
            while i < j and mul >= k:
                mul /= nums[i]
                i += 1
            # 这里跳出了while循环, 则可以把[i,j]窗口加入res
            res += (j-i)
    return res
nums, k = [10,5,2,6], 100
print('乘积<k的连续子数组, 有多少对: ', sub_bins_less_target(nums, k))

# 所有奇数长度子数组的和
def fun(arr):
    # 先遍历一遍数据把前缀和都算好, 
    # 第二次再遍历, [i, j] j每次间隔2来取, 则实现了奇数长度
    n = len(arr)
    pre_sum = [0]*(n+1)
    for i in range(n):
        pre_sum[i+1] = pre_sum[i] + arr[i]
    
    # ij双层遍历
    res = 0
    for i in range(n):
        for j in range(i, -1, -2):
            res += pre_sum[i+1] - pre_sum[j]
    return res 
arr = [1,4,2,5,3]
print('所有奇数长度子数组的和', fun(arr))

# 双指针 数组
# 最大连续1的个数: 最多可把k个0转为1, 然后求连续1的最长值
def longestOnes(nums, k):
    res, l, r = 0, 0, 0
    while l < len(nums) and r < len(nums):
        if nums[r] or k: # k还没减到0, 或者r位置处本身就是1
            if not nums[r]: # r处是0那只能牺牲k减去1次
                k -= 1
            # r放心后移, 1个数的累加在r+1这体现
            r += 1
        else:  # nums[r]==0 and k ==0
            # 左边界是0, 那就让r右移, 舍弃掉这个l保持r-l不变
            if nums[l] == 0:
                r += 1
                # 如果不是左边界==0, 那r右移也无意义. 故r+=1写在if内
            l += 1
        res = max(res, r-l)
    return res 
nums, k = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3
print('最多可把k个0转为1, 然后求连续1的最长值: ', longestOnes(nums, k))

# 最长连续/上升子序列 (数值上连续上升, 元素的index随意)
# 哈希表, 记录各个元素参与的组合,的连续长度值. O(n)遍历数组, 出现重复值跳过不处理即可. (只管num not in dict 即可.)
# dict中num+-1的value直接累加到连续长度值上. 
def longestConsecutive(nums):
    dict_ = dict()
    mmax = 0
    for num in nums:
        if num not in dict_:
            letf = dict_.get(num-1, 0)
            right = dict_.get(num+1, 0)
            cur_lens = letf + right + 1
            mmax = max(mmax, cur_lens)
            # 更新当前元素的连续长度值
            dict_[num] = cur_lens
            # 把总连续长度, 赋值更新到所属组合的左右端点上.
            dict_[num-letf] = cur_lens
            dict_[num+right] = cur_lens
        # else: 
        #     # 数组重复的num直接不处理跳过
    return mmax
nums = [0,3,7,2,5,8,4,6,0,1] # [100,4,200,1,3,2]
print('数组中数值连续, index无要求的, 最长连续值: ', longestConsecutive(nums))

# 四数相加 4数相加  哈希表
# nums1,2,3,4各取一个数使得4个数之和为0. 则ab先组合出所有可能和的dict, 然后再遍历cd, 出现-(a+b)值则res加上value.
def fourSumCount(nums1, nums2, nums3, nums4):
    res = 0
    ab_map = dict()
    for a in nums1:
        for b in nums2:
            ab_map[a+b] = ab_map.get(a+b, 0)+1
    for c in nums3:
        for d in nums4:
            if -c-d in ab_map:
                res += ab_map[-c-d]
    return res 
nums1, nums2, nums3, nums4 = [1,2], [-2,-1], [-1,2], [0,2]
print('四个数组各取一个值, 求和等于0的组合数: ', fourSumCount(nums1, nums2, nums3, nums4))

# n长度的数组, 包含1~n的数值. 有些数重复则有些数没出现, 求没出现/消失的数值.
def findDisappearedNumbers(nums):
    # 消失的数的原因: 有些元素重复, 
    # 则以这些元素为index去对一个新的数组赋值, 新数组中位置为空的index则为消失的数
    n = len(nums)
    new_nums = [0]*n
    for num in nums:
        new_nums[num-1] = 1
    # 这一行需要细心点.
    return [ind for ind in range(1, n+1) if new_nums[ind-1] == 0] 
nums = [1,1] # [4,3,2,7,8,2,3,1]
print('长度为n, 数值范围1-n. 求消失的数字: ', findDisappearedNumbers(nums))

# 贪心算法, 排序   用最少数量的箭引爆气球
# 根据右边界把会重合的子数组合并起来, 
def findMinArrowShots(points):
    lens = len(points)
    if lens < 1:
        return 0
    # 按右边界升序
    points = sorted(points, key=lambda x: x[1], reverse=False)
    res = 1
    cur_ = points[0][1]
    for i in range(1, lens):
        # 左侧的右边界无法包含后面的左边界了, 则射箭+1且更新cur_
        if cur_ < points[i][0]:
            res += 1
            cur_ = points[i][1]
    return res  
points = [[10,16],[2,8],[1,6],[7,12]]
print('最少射箭次数: ', findMinArrowShots(points))

# 安排会议, 双指针, 数组, 排序
# 左边界排序  [其实不用限制左边界排序, 因为会议时间都满足e>s, so直接sorted即可~]
def get_meeting(slots1, slots2, duration):
    # 先数组排序
    slots1 = sorted(slots1)
    slots2 = sorted(slots2)
    l1, l2 = 0, 0
    while l1 < len(slots1) and l2 < len(slots2):
        meet_s, meet_e = max(slots1[l1][0], slots2[l2][0]), min(slots1[l1][1], slots2[l2][1])
        if meet_e - meet_s >= duration:
            return [meet_s, meet_s+duration]
        else:
            # 不需要考虑slots1之间的01交错比, [1]之间的比较可直接决定l1+1orl2+1
            if slots2[l2][1] >= slots1[l1][1]:
                l1 += 1
            else:  # slots2[lw][1] < slots1[l1][1]
                l2 += 1
    return []
slots1, slots2, duration  = [[10,50],[60,120],[140,210]], [[0,15],[60,70]], 12   
print('安排会议起始时间: ', get_meeting(slots1, slots2, duration))      

# 两个行程编码数组的积  双指针
def findRLEArray(encoded1, encoded2):
    ans = []
    i, j = 0, 0
    while i < len(encoded1) and j < len(encoded2):
        s = encoded1[i][0] * encoded2[j][0]
        # ans是空的则直接把s放入, 
        # or ans的最后元素不是s, 也把s放入并开始为s计数
        if not ans or ans[-1][0] != s:
            ans.append([s, 0])
        min_num = min(encoded1[i][1], encoded2[j][1])
        # 给刚积累到的s乘积值累计数字
        ans[-1][1] += min_num
        # 更新ij值 
        if encoded1[i][1] == min_num:
            i += 1
        else:
            encoded1[i][1] -= min_num
        if encoded2[j][1] == min_num:
            j += 1
        else:
            encoded2[j][1] -= min_num
    return ans 
encoded1, encoded2 = [[1,3],[2,1],[3,2]], [[2,3],[3,3]]
print('两个行程码的积: ', findRLEArray(encoded1, encoded2))

# 接雨水
def trap(height):
    ans = 0
    h1 = 0
    h2 = 0
    for i in range(len(height)):
        h1 = max(h1,height[i])
        h2 = max(h2,height[-i-1])
        ans = ans + h1 + h2 - height[i]
    return  ans - len(height)*h1
height = [0,1,0,2,1,0,1,3,2,1,2,1]
print('接雨水: ', trap(height))


# 字符串 
# 字符串排序, 一个很新很牛的做法. 可以把字符相同但组合不同的所有strings转化为一种string, 则可以比较是否满足变位词了. 
# 和把各个字符转为位运算组合起来, 异曲同工. (单词长度的最大乘积)
# 变位词  s2中是否包含s1的变位词. 变位词: 单词可以换顺序,但是不能中间夹杂别的字母.
def checkInclusion(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l1 > l2:
        return False
    # 对string做排序, 可保证有相同字符集的各种string得到一致结果
    s1 = ''.join(sorted(list(s1)))
    # 查看各个s2[i, i+l1+1]的排序是否与s1的排序一致
    for i in range(l2+1-l1):
        sub_s2 = ''.join(sorted(list(s2[i:i+l1])))
        if sub_s2 == s1:
            return True
    return False
# ord()+位运算做法不行, 如: hello和heloo的结果一致. 它保证的是字符种类的一致, 没有严格到各个字符个数也完全一致.
# 不过可对字符种类和个数做统计, a~z分别放在0~25index做计数统计, 得到的数字一致则也表示属于变位词
def count_char(word):
    count_ = [0]*26
    for char in word:
        count_[ord(char)-97] += 1
    return count_
def checkInclusion(s1, s2):
    l1, l2 = len(s1), len(s2)
    if l1 > l2:
        return False
    s1_char_count = count_char(s1)
    # 查看各个s2[i, i+l1+1]的排序是否与s1的排序一致
    for i in range(l2+1-l1):
        sub_s2 = count_char(s2[i:i+l1])
        if sub_s2 == s1_char_count:
            return True
    return False
s1, s2 = "hello", "ooolleoooleh"
print('字符串排序实现, 完全一样的字符组合排序不同, 但结果值一致, 实现变位词查找: ', checkInclusion(s1, s2))

# 字符串中的所有变位词 (输出全部组变位词组合, 返回对应在s2中的初始index)
# 延用上题的方案2, 把ord(char)-97把a~z转为0~25, 作为索引对字符计数.
# 并且滑窗思想, 固定窗内计数数组一致则可以是变位词.
def findAnagrams(s2, s1):
    l1, l2 = len(s1), len(s2)
    if l1 > l2:
        return []
    # 统计s1, s2两个单词, 分别含有26个字母的个数情况
    char1, char2 = [0]*26, [0]*26
    res = []
    # 做s1的字符个数统计, 顺便做s2的首个子段字符个数统计
    for i in range(l1):
        # ord('a') = 97
        char1[ord(s1[i])-97] += 1
        char2[ord(s2[i])-97] += 1
    if char2 == char1:
        res.append(0)
    # 接下来统计s2中剩余子段的字符统计情况
    for i in range(l2-l1):
        # 滑动窗, 右边统计一个, 左边减少一个
        char2[ord(s2[i+l1])-97] += 1
        char2[ord(s2[i])-97] -= 1
        # 每滑动一次都比较一次, 是否新增了相等的.
        if char2 == char1: # 以0为起点的单独比过了, 所以i的基础上+1
            res.append(i+1)
    return res
s1, s2 = 'abc', "cbaebabacd"
print('所有变位词: ', findAnagrams(s2, s1))        

# 滑动窗: 一边不断r+1扩大窗, 一边遇到重复元素则不断右移左边界缩窗
# 最长不重复连续子串 
def lengthOfLongestSubstring(nums):
    lens = len(nums)
    # 维护不定长滑动窗winds, 一边r不断往右扩大窗
    # 一边在出现重复元素的时候, 左边界往右内缩, 去掉前面的所有重复子list
    l, r, res = 0, 0, 0
    winds = []
    for r in range(lens):
        while l<r and nums[r] in winds:
            # 每次往右缩一位l+=1直到这个winds内无重复元素了, 新的无重复wind的起点l也就到位了~
            winds = winds[1:]
            l += 1
        # 这里不管上面的while, 就是直接的不断把r+1外扩窗
        winds.append(nums[r])
        res = max(res, r-l+1)
    return res 
nums = "pwwkew"
print('滑动窗求解, 最长的不含重复元素的连续子序列: r++,l--: ', lengthOfLongestSubstring(nums))

# 滑动窗内的最大值, 返回list 每个滑动范围内的最值依次输出.
def maxSlidingWindow(nums, k):
    win, res = [], []   
    for i, v in enumerate(nums):
        # i-k,i两个边界,达到长度k, 左小右大, 则需要pop掉win的第一个元素 
        if i >= k and win[0] <= i - k:   
            win.pop(0)   
        # win[-1]为窗内的最尾元素, 其对应的元素值小于等于当前滑到的值, 则更新wind的最尾index   
        while win and nums[win[-1]] <= v:
            win.pop()
        win.append(i)  
        if i >= k - 1: # i>=k-1表示i滑动到满足k大小, 可开始取窗内的数值了 
            # 每次是append wind[0]处的值, 因为wind[-1]可能对应后续滑窗的最值
            res.append(nums[win[0]])  
    return res
nums, k = [1,3,-1,-3,5,3,6,7], 3
print('滑动窗内的最大值: ', maxSlidingWindow(nums, k))

# 字符串, 回文串
# 比较简单, 属于数学找规律. 字符串可以任意换顺序, 只要个数上满足回文要求即可
def longestPalindrome(s):
    # 哈希表{char:value}
    dict_ = dict()
    for char in s:
        dict_[char] = dict_.get(char, 0) + 1
    res = 0
    ji_flag = 0
    # 回文的中心只可能<=1个单独的字符, 故统计每个字符出现的个数并//2,
    # ji_flag记录是否有出现奇数次的字符, 有的话则可以取一个放到回文的中心
    for char, value in dict_.items():
        if value % 2 == 1:
            ji_flag = 1
        res += value//2
    return res*2+ji_flag

# 回文进阶: 
# 不可改变string的顺序, 找出最长的回文子串
# dp[i][j]维护i~j内满足回文串的bool值, false or true
def longestPalindrome(s):
    # i: 0~j, j: 0~lens
    l = len(s)
    if not l:
        return s
    tmp = 0
    res = ''
    dp = [[0]*l for i in range(l)]
    for j in range(l):
        for i in range(j, -1, -1):
            if s[j]==s[i] and (j-i<2 or dp[i+1][j-1]):
                # 当扫描到ij元素相等, 要么ij相邻(j-i<2)
                # 要么dp[j+1][i-1]已经满足了==1是回文
                # 这两种情况才能在s[i]==s[j]下更新dp[i][j]=1
                dp[i][j] = 1
            if dp[i][j] and tmp < j-i+1:
                tmp = j-i+1
                res = s[i:j+1]
    return res 
s = 'babad'
print('最长回文串, dp实现: ', longestPalindrome(s))

# 最多可删除一个字符, 实现回文. (不可改变字符顺序)
# 递归做法
def reverse_(l,r,s):
    while l <= r and s[l] == s[r]:
        l += 1
        r -= 1
    return r <= l 
def validPalindrome(s):
    lens = len(s)
    slim_flag = 1  # 可使用一次的删除功能
    i,j = 0, lens-1
    while i <= j and s[i] == s[j]:
        i += 1
        j -= 1
    return reverse_(i+1, j, s) or reverse_(i, j-1, s)
s = 'abc'
print('最多删除一个字符得到回文: ', validPalindrome(s))

# dp 回文字符串个数
def countSubstrings(s):
    lens = len(s)
    count = 0
    dp = [[False]*lens for _ in range(lens)]
    # i从后往前, 判断[i,j]是否为回文
    for i in range(lens-1, -1, -1):
        for j in range(i, lens):
            # j-i<=1 or dp[i+1][j-1] 这个顺序很巧妙, 
            # j-oi<=1在前, 先保证ij是否为同一个元素or相邻, 则只需要结合前面的s[i]==s[j]即可
            # 当走到后面的dp[i+1][j-1], 说明j-i>1, 则[i+1][j-1]也就不用担心过界
            if s[i] == s[j] and (j-i<=1 or dp[i+1][j-1]):
                dp[i][j] = True
                count += 1
    return count
s = 'aaa'
print('共有多少个回文子串: ', countSubstrings(s))

# 通配符匹配 hard 
'''
'?' 可以匹配任何单个字符
'*' 可以匹配任意字符串(包括空字符串)
'''
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True
    # init
    for i in range(n):
        # p的前一位已匹配dp值为1,且p的当前位是*万能匹配, 则dp[0][i+1]也可置1
        if dp[0][i] and p[i] == '*':
            dp[0][i+1] = True
    for i in range(m):
        for j in range(n):
            if p[j] == '*':
                dp[i+1][j+1] = dp[i+1][j] or dp[i][j+1]        
            elif p[j] == '?' or s[i] == p[j]:
                dp[i+1][j+1] = dp[i][j]
    return dp[-1][-1]
print('?*是否可匹配: ', isMatch('cb', '?a'))

# 最短回文串: 在string的首部添加一些字符使得变成回文串
def shortestPalindrome(s):
    lens = len(s)
    if not lens:
        return ''
    # 逆序
    rs = s[::-1]
    # 截取逆序后的前i个字符拼接到s上
    i = 0
    while True:
        # s[:lens-i]原s的前lens-i子串
        # rs[i:]: 对应原s的前lens-i个的逆序, 
        # 更新这个i, 就是把原s中可自行满足回文的长度统计出来
        if rs[i:] == s[:lens-i]:
            break
        i += 1
    # 跳出循环了, 则找到了lens-i个可自行回文的sub_s
    # 想拼接成完整回文, 则需要补的长度: lens-(lens-i) = i
    # 且直接把re的前i位搬过来就好
    return rs[:i] + s
s = 'abcd'
print('在string前面补充最短的字符,实现string变位回文串: ', shortestPalindrome(s))

# 哈希表 滑动窗 字符串
# 含有所有字符的最短连续子串  hard  最小覆盖子串
def minWindow(s, t):
    l1, l2 = len(s), len(t)
    if l1 < l2 or l2 == 0:
        return ''
    # j位窗口右边界, cnt为窗口内t的不同元素个数
    # ans记录结果的左边界和结果长度
    j, cnt, ans = 0, 0, [-1,-1]
    dict_t = dict()
    for _ in t:
        dict_t[_] = dict_t.get(_, 0)+1
    t_num = len(dict_t)
    # i为左边界
    dict_s = dict()
    for i in range(l1):
        # 持续外扩右边界,到满足元素种类可退出while
        while j < l1 and cnt < t_num:
            dict_s[s[j]] = dict_s.get(s[j], 0) + 1
            if s[j] in dict_t and dict_t[s[j]] == dict_s[s[j]]:
                cnt += 1
            j += 1
        # t中的元素都找全了, 可以开始内缩左边界了
        if cnt == t_num:
            # ans[0]==-1表示第一次更新. (没有这个的话, 后面的ans[1] > j-i不会被满足, 因为ans[1]给的初始值是-1)
            # ans[1] > j-i条件则是: 此时得到的j-i窗口大小比之前的ans[1]小, 则更新为更小的窗口.
            if ans[0] == -1 or ans[1] > j-i:
                ans = [i, j-i]
        # 判断内缩的i是否则t中有, 则做对应计数变化
        if s[i] in dict_t and dict_t[s[i]] == dict_s[s[i]]:
            cnt -= 1
        dict_s[s[i]] -= 1  # i自动+1因为在for循环内哈~
    return '' if ans[0] == -1 else s[ans[0]:sum(ans)]
s, t= "ADOBECODEBANC", "ABC"
print('最短的包含全部字符的长度: ', minWindow(s, t))


a = {'s': 1, 'c': 2}
b = {'c': 2}
print(a['c'] == b['c'])




############################################################
# 位运算 
# 整数除法
def divide(a,b):
    res = 0
    flag = -1 if (a*b<0) else 1
    a, b = abs(a), abs(b)

    while a >= b:
        # 返回二分能积累到的最大商, 和此时对应的big_b
        base_res, big_b = 1, b
        while a > (big_b<<1):
            big_b <<= 1
            base_res <<= 1
        # 退出的情况是: a <= big_b<<1
        res += base_res
        a -= big_b
        # 到这里res加好了已有的商, 也减去了big_b, 再来跟原始的b比较
        # <b则退出函数, 大于b的话则再重新进入第一层while. a变小了, b不变哇~
    res *= flag 
    return res - 1 if res >= 2 ** 31 else res
a,b =7, -3
print('不用除号运算做整数除法: {}/{}={}'.format(a,b,divide(a,b)))

# 二进制加法
def addBinary(a, b):
    # 置换a,b, 让a是更长的那个, 则按照a的长度左侧补齐0
    n1, n2 = len(a), len(b)
    if n1 < n2:
        a,b = b,a
        n1 = n2
    b = b.rjust(n1,'0')
    # 原地ab相加 结果给到a上, 后面a的每一位跟2比较就好
    a = [int(a[i])+int(b[i]) for i in range(n1)]
    
    # a的左侧加一个1, 一会可能要进位.
    # 两数之和肯定只会<=最大数的两倍, so补一位即可. 左移1位就是*2了.
    a = [0] + a
    # n1其实是a现在的长度n1+1再减去1; 第二个0则是不处理a的第一位进位位.
    for i in range(n1, 0, -1):
        if a[i]>1:
            a[i] -= 2
            # 进位位+1
            a[i-1] += 1
    # 第一位没有进位到的话, 把他剔除
    if not a[0]:
        a =  a[1:]
    return ''.join(str(_) for _ in a)
a, b = '1', '111'
print('二进制加法: {}+{}={}'.format(a,b,addBinary(a,b)))

# 前 n 个数字二进制中 1 的个数
# 其实就是[0,n]各个数字的二进制表示, 有几个1
def countBits(n):
    res = [0]*(n+1)
    for i in range(n+1):
        # 右移一位==除以2
        # 再看i这位是否会在/2的基础上,新增1: i&1
        res[i] = res[i>>1] + (i&1)
    return res 
n = 5
print('前 n 个数字二进制中 1 的个数', n, countBits(n))   

# 只出现一次,其他均出现3次的数字
def singleNumber(nums):
    a, b = 0, 0
    for num in nums:
        # x&x=x, x^x=0, 0^y=y so最后会剩下那个一次的数
        a ^= num
        b ^= (a&num)
        a ^= (b&num)
    return b 
nums = [0,1,0,1,0,1,100]
print('均出现三次只ta出现一次的数: {}'.format(singleNumber(nums)))

# 单词长度的最大乘积
def maxProduct(words):
    # 字符转为二进制ord(), 与运算, 为0表示不同.
    lens = len(words)
    arr = []
    for i, word in enumerate(words):
        # 每次单词,先初始化0 
        arr.append(0)
        for char in word:
            # ord('a') = 97, 每个字母都减去a这个基准, 得到: 0~25数值
            # 并且左移1位直接变a^(0~25)
            # 每个单词以0为起点, |= 异或不断积累各个字符过来的值
            arr[i] |= 1 << ord(char) - 97
    # 接下来就是找, arr中元素不等, 且长度最大的两个词.
    res = 0
    for i in range(lens-1):
        for j in range(i+1, lens):
            # 与处理等于0则为不等, 即不含相同字符的word
            if arr[i] & arr[j] == 0:
                len_mm = len(words[i])*len(words[j])
                res = len_mm if len_mm > res else res 
    return res 
words = ["abcw","baz","foo","bar","fxyz","abcdef"]
print('不含相同字符, 且长度乘积最大的: ', maxProduct(words))


############################################################
# 动态规划
# 斐波那契数
def fib(n):
    fn = [0,1,1]
    if n <= 2:
        return fn[n]
    a, b = 0,1
    # 为什么是n-1? 因为a,b已经占了俩数, so剩下的计算只需要进行n-2次, 也即range(n-1)
    for i in range(n-1):
        c = a+b
        a, b = b, c
        fn.append(c)
    return fn[-1]
## dp写法:
def fib(n):
    fn = [0,1,1]
    if n <= 2:
        return fn[n]
    fn = [0]*(n+1)
    fn[:3] = [0,1,1]
    for i in range(2, n+1):
        fn[i] = fn[i-1]+fn[i-2]
    return fn[-1]
print('斐波那契数: ', n, fib(n))

# 复杂版爬楼梯:
'''
输入: cost = [1,100,1,1,1,100,1,1,100,1]
输出: 6, 可选i+1 or i+2处去爬, 花费对应费用.

'''
def minCostClimbingStairs(cost):
    n = len(cost)
    # dp: 爬到n位置需要的最小费用, 没算到达当前位置的费用.
    dp = [0]*(n+1)
    for i in range(2, n+1):
        dp[i] = min(dp[i-2]+cost[i-2], dp[i-1] + cost[i-1])
    return dp[-1]

# 左上角走到右下角, 可以有的多少种可能? 每步可右or下
def uniquePaths(m, n):
    dp = [[0]*n for i in range(m)]

    # 初始化第一行和第一列, 都是1.
    dp[0] = [1]*n
    for j in range(m):
        dp[j][0] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]
m, n = 3,7
print('左上到右下, 向下或向右, 路径可能:', uniquePaths(m, n))

# 有障碍物, 其他和上题一致
def uniquePaths_(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for i in range(m)]
    # 这里需要给[0][0]位置1初始化, 后面会用到
    dp[0][0] = 1

    # 终点有障碍物, 则无法到达
    if grid[-1][-1] == 1:
        return 0 

    for i in range(m):
        for j in range(n):
            # i,j位置可达, 才需要更新此处的dp值
            if grid[i][j] != 1:
                # grid[i-1][j] != 1保证此处没堵住
                if i >= 1 and grid[i-1][j] != 1:
                    dp[i][j] += dp[i-1][j] 
                if j >= 1 and grid[i][j-1] != 1:
                    dp[i][j] += dp[i][j-1]  
    return dp[-1][-1]
grid = [[0,0,0],[0,1,0],[0,0,0]]
print('有障碍, 上到右下路径可能:', uniquePaths_(grid))

# 左上到右下, 最小路径和值
def minPathSum(grid):
    if not grid or not len(grid[0]):
        return
    m,n = len(grid), len(grid[0])
    dp = [[0]*n for i in range(m)]
    dp[0][0] = grid[0][0]
    # init 
    for i in range(1, m):
        dp[i][0] = grid[i][0] + dp[i-1][0]
    for j in range(1, n):
        dp[0][j] = grid[0][j] + dp[0][j-1]

    for i in range(1,m):
        for j in range(1,n):
            dp[i][j] = min(dp[i-1][j], dp[i][j-1])+grid[i][j]
    return dp[-1][-1]

# 

############################################################
# 数组, 二分查找, 双指针(使用场景一般是,这个list是有序的了)
# 升序数组,最左开始的第k个缺失的数 
def missingElement(nums, k):
    # 截止到index位置缺失数的个数: nums[ind]-nums[0]-ind
    l, r = 0, len(nums)-1
    while l <= r:
        mid = (l+r)//2
        missed = nums[mid]-nums[0]-mid
        if missed >=k:
            r = mid-1
        else:
            l = mid+1
    return k+nums[0]+r

# 二维数组中的峰值, num大于上下左右,即认为是一个峰值
def findPeakGrid(mat):
    m = len(mat)
    # 只在行维度上做二分, 行内的max_index直接调用了py的函数.
    l, r = 0, m -1
    while l<= r:
        mid = (l+r)//2
        mid_max_index = mat[mid].index(max(mat[mid]))
        # mid行的最大值不满足大于上一行index处的值, 则mid需要向上调
        if mid >= 1 and mat[mid][mid_max_index]<mat[mid-1][mid_max_index]:
            r = mid-1
        elif mid < m-1 and mat[mid][mid_max_index]<mat[mid+1][mid_max_index]:
            l = mid+1
        else:
            return [mid, mid_max_index]
    return [-1, -1]

# 二分 O(1)空间复杂度, 找出n+1长度数组中的唯一一个重复元素 value范围:1~n
def findDuplicate(nums):
    # 二分查找, 看l~mid之间的个数与mid值的大小关系, 
    # 个数>mid值则有重复, 个数<mid值则在mide的另一侧重复,把l调大吧.
    # n+1个数, 范围再1~n, 只有一个重复的数
    lens = len(nums)
    l, r = 1, lens
    while l<r:
        mid = (l+r)//2
        cnt = 0  # 统计<=mid的个数
        for num in nums:
            if num <= mid:
                cnt += 1
        if cnt <= mid:
            l = mid + 1
        else:
            r = mid 
    return r 
nums = [3,1,3,4,2]
print('n+1数组内且范围1-n, 唯一重复的元素: ', findDuplicate(nums))

# 分巧克力
def splitCnt(min_sweet, sweetness):
    cnt, partsum = 0, 0
    for sweet in sweetness:
        partsum += sweet 
        if partsum >= min_sweet:
            cnt += 1
            partsum = 0
    return cnt 

def maximizeSweetness(sweetness):
    sum_sweet = sum(sweetness)
    min_sweet = min(sweetness)
    l, r = min_sweet, sum_sweet
    while l <= r:
        mid = (l+r)//2
        cnt = self.splitCnt(mid, sweetness)
        if cnt >= k+1:
            l = mid+1
        else:
            r = mid-1
    return min_sweet if r < min_sweet else r 

# 从索引i到颜色c需要的最短距离
def shortestDistanceColor(colors, queries):
    # 正逆序各一遍,得到每个index的最近123颜色距离. min(左,右)
    n = len(colors)
    dp = [[-1,-1,-1] for _ in range(n)]  # [n][3]
    distance = [-1,-1,-1]
    for i in range(n):
        distance[colors[i]-1] = i 
        dp[i][colors[i]-1] = 0
        for idx, j in enumerate(distance):
            if j != -1:
                dp[i][idx] = abs(i-j)
    for i in range(n-1, -1, -1):
        distance[colors[i]-1] = i 
        for idx, j in enumerate(distance):
            if j != -1:
                if dp[i][idx] != -1:
                    dp[i][idx] = min(abs(i-j), dp[i][idx])
                else:
                    dp[i][idx] = abs(i-j)
    res = []
    for q in queries:
        res.append(dp[q[0]][q[1]-1])
    return res 
colors, qus = [1,1,2,1,3,2,2,3,3], [[1,3],[2,2],[6,1]]
print('从制定索引到指定颜色的距离: ', shortestDistanceColor(colors, qus))


# 三个数之和等于0的组合
def threeSum(nums):
    lens = len(nums)
    res = []
    nums = sorted(nums)
    for i in range(lens-2):
        if i >= 1 and nums[i-1] == nums[i]:
            continue
        # [i,j,k]
        j, k = i+1, lens-1 
        while j < k:
            # i,j,k三个和为0
            sum_ = nums[j]+nums[k]
            if -nums[i] == sum_:
                res.append([nums[i], nums[j], nums[k]])
                # 和满足==0了, 就可以把jk重复的都移位处理掉
                while j < k and nums[j] == nums[j+1]:
                    j += 1
                j += 1
                while j < k and nums[k] == nums[k-1]:
                    k -= 1
                k -= 1
            elif sum_ > -nums[i]:
                k -= 1
            else:
                j += 1
    return res 
nums = [-1,0,1,2,-1,-4]
print('三数之和等于0, 的组合: ', threeSum(nums))


############################################################
# 链表 
# 删除链表的倒数第n个节点: qp双指针,p先走n到结尾,此时q即为倒数第n.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
def removeNthFromEnd(head):
    # 双指针在链头, p先走n步,再pq一起走. 
    # 当p到达表尾, q则正处于倒数第n个节点.
    p,q,o = head,head,head
    for i in range(n):
        p = p.next
    while p:
        p = p.next
        # o实时记录q的前一个节点, 一会倒数n会用上它
        o = q
        q = q.next
    # p到尾了, 则q正好在倒数n
    if q == head:
        head = head.next
        return head
    # 删除掉了q节点
    o.next = q.next
    return head

# 链表的环入口
# 也是双指针, 快慢走. 
def detectCycle(head):
    # 快慢指针+1,+2走, 相遇时慢指针走的步数则为环的长度.
    # 更抽象"简单"的做法是: 找到被重复遍历的节点, 返回即可.
    node = head
    set_ = []
    while node:
        if node not in set_:
            set_.append(node)
        else:
            return node 
        node = node.next
    return None  

# 排序的循环链表, 插入一个点使仍有序, 返回插入的index
def insert(head, insertVal):
    if not head:
        node = Node(insertVal)  # 用插入值生成一个新node
        node.next = node  # 循环链表是这样的!
        return node
    cur = head 
    while cur.next.val >= cur.val:
        cur = cur.next
        if cur == head: break 
    small = cur.next
    p, pre = small, cur  
    while p.val < insertVal:
        pre = p 
        p = p.next
        if p == small: break
    node = Node(insertVal)
    pre.next = node
    node.next = p 
    return head 
print('有序的循环链表, 插入一个值仍有序.')

# 实现一个插入, 删除, 随机访问都是O(1)的容器
class RandomizedSet:
    def __init__(self):
        self.lst = []  # 维护所有插入元素
        self.dct = dict()  # 记录插入元素的index
    def insert(self, val: int) -> bool:  # 插入 
        if val in self.dct:  # 已经包含此元素则无需再插入
            return False
        self.lst.append(val)  # 插入最尾部, O(1)
        self.dct[val] = len(self.lst) - 1  # 因是插入在最尾,so index是lens-1
        return True
    def remove(self, val: int) -> bool:  # 删除 
        if val not in self.dct:
            return False
        i = self.dct[val]  # 要删除的值val对应index=i 
        self.lst[i] = self.lst[-1]  # O(1)故得从最尾删, 把最为元素换过来"替罪"
        self.dct[self.lst[i]] = i  # 更新上一步替代元素对应的新index,为i
        self.lst.pop(-1)  # 删最尾元素 
        self.dct.pop(val)  # 删除val这个key
        return True
    def getRandom(self) -> int:
        return self.lst[random.randint(0, len(self.lst) - 1)]

# 最近最少使用缓存
import collections
class LRUCache(collections.OrderedDict):
    # 插入 变更 删除最历史数据保持len不变
    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self:
            return -1
        self.move_to_end(key)
        return self[key]

    def put(self, key: int, value: int) -> None:
        if key in self:
            self.move_to_end(key)
        self[key] = value
        if len(self) > self.capacity:
            self.popitem(last=False)
        return


# 两个链表的第一个重合节点
def getIntersectionNode(headA, headB):
    # 先ab一起走, 短的会先到尾, 然后把指针指向长的的表头, 再一起走, 
    # (这就实现了计算出两条差多少并且让长的那条先走了差距步)
    # 两个指针相遇处, 即为交点处
    p, q = headA, headB
    while q != p:
        p = p.next if p else headB
        q = q.next if q else headA
    return p  # return q也是一样的, 此时相遇在交点处

# 复制带随机指针的链表
def copyRandomList(head):
    def copyNode(node, dict_):
        # 因为有random,so可能出现node重复指向,用dict直接包了
        if not node: return None
        if node in dict_: return dict_[node]
        copy = Node(node.val, None, None)  # 复制val,next,random信息
        dict_[node] = copy 
        copy.next = copyNode(node.next, dict_)
        copy.random = copyNode(node.random, dict_)
        return copy
    return copyNode(head, {})

# 扁平化多级双向链表: pre,next, child 压缩链表维度至1维
def flatten(head):
    curr = head
    while curr:
        if curr.child:  # curr节点有node,则做处理删掉
            real_next = curr.next
            child = curr.child
            # 有child,则先在curr开一个口, 然后把child放这个口上
            curr.next = child  # child真实值放到curr的next了
            curr.child = None  # 消除child指针
            child.prev = curr  # 对应curr.next = child~
            # 找到当前child链的最末尾, 把他接到curr的最初next的前一个
            while child.next:
                child = child.next
            if real_next:
                real_next.prev = child  # 衔接上了~
            child.next = real_next
        curr = curr.next  # curr继续往后遍历
    return head

# 链表, 双指针, 数学
# 求两个多项式链表的和
def addPoly(poly1, poly2):
    p1, p2, ans_tail = poly1, poly2, None
    ans = []
    while p1 or p2:
        if p1 == None or (p2 and p2.power > p1.power):
            cur_coff, cur_pow = p2.coefficient, p2.power
            p2 = p2.next
        elif p2 == None or (p1 and p1.power > p2.power):
            cur_coff, cur_pow = p1.coefficient, p1.power
            p1 = p1.next
        else:
            cur_pow, cur_coff = p1.power, p1.coefficient + p2.coefficient
            p1, p2 = p1.next, p2.next
    
        if cur_coff != 0:
            newNode = PolyNode(cur_coff, cur_pow, None)
            if not ans:
                ans, ans_tail = newNode, newNode
            else:
                ans_tail.next = newNode
                ans_tail = ans_tail.next

    return ans

# 给链表加1: 难点在,可能会进位, 则会影响到前面的节点
def plusOne(head):
    cur, pre = head, None
    while cur:
        temp = cur.next
        cur.next = pre
        cur, pre = temp, cur
    # 反转后做+1
    cur1 = pre
    k = 1
    while k:
        cur1.val += 1
        if cur1.val == 10:
            cur1.val = 0
            if cur1.next:
                # 可进位不用申请多一个节点
                cur1 = cur1.next
                k = 1
            else:
                cur1.next = ListNode(1)
                k = 0
        else:
            k -= 1
    # 反回来
    cur2, ans = pre, None
    while cur2:
        temp = cur2.next
        cur2.next = ans
        cur2, ans = temp, cur2
    return ans 

# 合并两个有序链表
def mergeTwoLists(list1, list2):
    # 合并为升序链表, tmp为设置的新链表头
    tmp = ListNode(0)
    phead = tmp
    while list1 and list2:
        if list1.val < list2.val:
            tmp.next = list1
            list1 = list1.next 
        else:
            tmp.next = list2
            list2 = list2.next
        tmp = tmp.next  # tmp也继续往后走
    # 跳出while, list1or2orboth遍历完了
    if list2:
        # list1遍历完了, 那么temp的next指到list2即可
        tmp.next = list2 
    if list1:
        tmp.next = list1 
    return phead.next
# 合并k个有序链表
def mergeKLists(lists):
    # 分治法, 两个两个的合并
    lens = len(lists)
    # 分治函数
    def merge(left, right):
        if left > right:
            return 
        if left == right:
            return lists[left]
        mid = (left+right)//2
        l1 = merge(left,  mid)
        l2 = merge(mid+1, right)
        return mergeTwoLists(l1, l2)
    # 合并俩链表
    def mergeTwoLists(l1, l2):
        if not l1 or not l2:
            return l1 or l2
        if l1.val < l2.val:
            l1.next = mergeTwoLists(l1.next, l2)
            return l1 
        else:
            l2.next = mergeTwoLists(l1, l2.next)
            return l2 
    # 实际就这一行代码
    return merge(0, lens-1)

# 链表中的两数相加: 先把俩链表都反转, 然后对应位相加, 有进位则往next位放
# 全部算完之后, 再反转回来   (因为链表的next否是往后走的, 但加法进位的原则都是往前走, so需要反转.) 
def reverse(self, head):  # 反转链表
    if not head:
        return None 
    pre, cur = None, head
    while cur:
        forward_cur_next = cur.next
        cur.next = pre
        pre = cur
        cur = forward_cur_next
    return pre 
def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
    # 先对l1,l2反转, 再对应为相加(有进位直接给到next节点上)
    l1 = self.reverse(l1)
    l2 = self.reverse(l2)
    res = ListNode()
    cur = res
    jinwei = 0
    while l1 or l2:
        num1 = l1.val if l1 else 0
        num2 = l2.val if l2 else 0
        sum_ = num1 + num2 + jinwei
        cur.next = ListNode(sum_%10)  # 相加结果赋值到结点上 
        jinwei = sum_//10
        l1 = l1.next if l1 else None
        l2 = l2.next if l2 else None 
        cur = cur.next
    if jinwei: # 加到最末尾还有一个进位值, 则再重新加一个尾巴结点
        cur.next = ListNode(1) 
    return self.reverse(res.next)

# 回文链表 翻转链表然后和原链表比较  easy
def isPalindrome(head):
    a = ''
    while head:
        a += str(head.val)
        head = head.next
    return a == a[::-1]

# 重排链表 双指针 递归
# l0->l1->...ln 变为: l0->ln->l1->ln-1->...
# 分步骤做: 1. 先快慢指针找中点(中点会放在最后);
# 2. 中点后部分做反转; 3. 中点前部分和反转的后部分, 逐节点穿插合并
def reorderList(head):
    pre = ListNode()
    pre.next = head 
    slow = fast = pre
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    half = slow.next   # 这个half会让后部分少一个 
    slow.next = None   # 在slow处截断, slow处结点为前半部的尾指针. 也就是截断到只剩前部分
    # 后半部分反转 
    rev_half = self.reverse(half)
    cur = pre.next  # 此时的cur是head
    while slow and rev_half:
        cur_forward_next = cur.next
        cur.next = rev_half  # 前穿插后
        cur = cur.next  # 前部分的值更新后移 
        rev_half = rev_half.next  # 后部分的值更新后移 
        # cur结点做后移遍历 
        cur.next = cur_forward_next
        cur = cur.next 

# 链表排序 归并排序 
def sortList(phead):
    if not phead:
        return None
    else:
        quicksort(phead,None) # head and end
    return phead
# p1 p2 节点交换
def swap(node1,node2):
    tem = node1.val
    node1.val = node2.val
    node2.val = tem
def quicksort(head,end):
    if head != end:
        key = head.val
        p = head
        q = head.next   # p q 两指针
        while q != end:  # q 遍历除参考值外的所有节点
            if q.val < key:  # 出现节点的值小于参考值
                p = p.next # 先把p前移一位，再给这个位置赋予刚刚q的值
                swap(p,q)# 将q的值给p  使得p遍历的节点都小于key
            q = q.next
        swap(head,p)  # 这一步别漏了，把key_ind和之前的head互换 然后分两段使两段均有序
        quicksort(head,p)
        quicksort(p.next,end)


###############################################################
# 递归 栈 字符串
# 字符串解码  
def decodeString(s):
    stack = []  # 栈内每次存两个信息, 左括号前的首字符,左括号后紧跟的数字 
    num = 0
    res = ''  # 实时当前可提取出来的字符串
    for c in s:
        if c.isdigit():  # 当前位数字
            num = num*10+int(c)
        elif c == '[':
            stack.append((res, num))
            res, num = '', 0
        elif c == ']':
            top = stack.pop()
            res = top[0]+res*top[1]  # (字符, 数字)
        else:
            res += c    
    return res
print('字符串解码: 3[a]2[bc]', decodeString('3[a]2[bc]'))



###############################################################
# 排序算法
# 冒泡排序
def maopao_sort(nums):
    # 最坏O(n^2) 最好:O(n)
    lens = len(nums)
    for i in range(lens):  
        # 持续把大的数放后面, j循环至lens-1-i, 因为后i个数已有序无序处理
        for j in range(lens-1-i):
            if nums[j] > nums[j+1]:
                nums[j], nums[j+1] = nums[j+1], nums[j]
    return nums
print('冒泡排序: ', maopao_sort([3,5,6,-1,10]))

# 快排   参考值, 左边放小的右边放大的
def quick_sort(nums, left, right):
    if left >= right:
        return nums
    low = left
    high = right
    key = nums[left]
    while left < right:
        while left < right and nums[right] >= key:
            right -= 1
        # 跳出上面的while，说明右边出现小于key的值，把它放到左边去
        nums[left] = nums[right]
        while left < right and nums[left] <= key:
            left += 1
        # 跳出上面的while，说明左边出现大于key的值，把它放到右边去
        nums[right] = nums[left]
    # 跳出循环，即left>=right
    nums[left] = key
    # 现在完成了小于等于key的在左，大于key的在右
    # 那就递归的把左右分别排序好吧
    # left处等于key值
    quick_sort(nums, low, left-1)
    quick_sort(nums, left+1, high)
    return nums
nums = [5,3,3,7,1,8,1,4]
print('快排: ', quick_sort(nums, 0, 7))

# 幂运算
def power(a, e):
    if a == 0 and e < 0:
        return 0
    if e == 0:
        return 1
    if e == 1:
        return a
    if e < 0:
        res = 1 / power(a, -e)    
        return res
    res = power(a, e >> 1) # 使用位运算右移,高效除以2
    res *= res  
    if e % 2 == 1:  # 指数为奇数则还需*a
        res *= a    
    return res
print(power(9.1, 2))

# 锯齿队列: 交替返回两个数组的头元素 
def __init__(v1,v2):
    self.arr = [v1, v2]  # arr存v1v2俩数组, 每次弹出arr[0]的头元素
    # 然后arr掉个个~
def next(self) -> int:
    if not self.arr[0]:
        self.arr = self.arr[::-1] # 逆序掉个
    res = self.arr[0].pop(0)
    self.arr = self.arr[::-1] # 逆序掉个
    return res 
def hasNext(self) -> bool:
    return not(self.arr[0]==self.arr[1]==[])  

# 每日温度  answer[i]:第一个温度大于第i天的day-index
def dailyTemperatures(temperatures):
    stack = [] # 维护一个递减栈,降序. 存的都是index
    lens = len(temperatures)
    res = [0]*lens
    for day, te in enumerate(temperatures):
        if stack: 
            while stack and temperatures[stack[-1]] < te:
                res[stack[-1]] = day - stack[-1]
                stack.pop() # [-1]比较完了pop掉
        stack.append(day)
    return res 
T = [73,74,75,71,69,72,76,73]
print('天气温度, 更高的温度出现在几天后? ', dailyTemperatures(T))


############################################################
# 深度优先dfs, 广度优先bfs
# leetcode286 墙与门  
def wallsAndGates(rooms):
    # bfs广度优先   0:门, -1:障碍物 INF:可通行
    move = [[-1,0],[1,0],[0, -1],[0,1]]  # 上下左右走
    m, n = len(rooms), len(rooms[0])
    # bfs 
    def bfs(i, j, rooms, count): # count为距离门的步数
        rooms[i][j] = min(rooms[i][j], count)  # 原地改变rooms值
        for r in range(4):
            x = i + move[r][0]
            y = j + move[r][1]
            # xy没有越出矩形边界, [xy]处不是门也不是墙,可行走.  
            # rooms[x][y] > count+1的原因: 应该是表示xy处没有更新到最佳的距离min(rooms[i][j], count). 
            if 0 <= x < m and 0 <= y < n and rooms[x][y] not in [0, -1] and rooms[x][y] > count+1:
                bfs(x,y, rooms, count+1)
        return 
    # main代码
    for i in range(m):
        for j in range(n):
            if rooms[i][j] == 0: # 找到门了, 开始bfs遍历, 完成门周边步数的更新
                bfs(i,j,rooms,0)
    return rooms
rooms = [[2147483647,-1,0,2147483647],[2147483647,2147483647,2147483647,-1],[2147483647,-1,2147483647,-1],[0,-1,2147483647,2147483647]]
print('门和墙: ', wallsAndGates(rooms))

# 水流问题 dfs
def shuiliu(heights):  # 大的值流向小的值
    m, n = len(heights), len(heights[0])
    def search(starts):  # starts: [(),(),..]
        visited = set()
        def dfs(x, y):
            if (x, y) in visited:
                return  # 遍历过了直接pass
            visited.add((x, y))    
            for nx, ny in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):  # 上下左右4个水的流动方向
                # nx,ny value>xy value (大的流向小的), 则需遍历nx,ny
                if 0 <= nx < m and 0 <= ny < n and heights[nx][ny] >= heights[x][y]:
                    dfs(nx, ny)
        # 这里是search的main代码部分
        for x, y in starts:
            dfs(x, y)
        return visited

    pacific = [(0, i) for i in range(n)] + [(i, 0) for i in range(1, m)]
    atlantic = [(m - 1, i) for i in range(n)] + [(i, n - 1) for i in range(m - 1)]
    return list(map(list, search(pacific) & search(atlantic)))
heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
print('水流问题, 太平洋大西洋, ', shuiliu(heights))

# 有向无环图所有可能路径   dfs: 一直走到底看看有无n-1这个value
def allPathsSourceTarget(graph):
    # graph的含义: 0~n-1各个数值结点, 可走向的数值
    res = []
    n = len(graph)-1  # n为要抵达的终点值
    def dfs(cur, path_):  # 当前走到的点的值, 已经走过的path_
        if cur == n:
            res.append(path_ + [cur])
            return 
        # graph[cur]为当前节点可走向的节点list
        for node in graph[cur]:
            dfs(node, path_+[cur])
    dfs(0, [])
    return res 
graph = [[4,3,1],[3,2,4],[3],[4],[]]
print('有向无环图所有可能路径: ', allPathsSourceTarget(graph))

# 树的独生节点
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right 
# 树的层次遍历list, 寻找无左or右兄弟的节点
# 1. 递归做法:
def getLonelyNodes(root):
    # 树的层次遍历list, 寻找无左or右兄弟的节点
    if bool(root.left or root.right) > bool(root.left and root.right):
        return [(root.left or root.right).val] + getLonelyNodes(root.left or root.right)
    return root.left and root.right and getLonelyNodes(root.left)+getLonelyNodes(root.right) or []
# 2. dfs做法  dfs本就是用递归思想来实现的
def getLonelyNodes(root):
    def dfs(root, ls):
        if not root:
            return 
        if root.left == None and root.right:
            ls.append(root.right.val)  
        if root.right == None and root.left:
            ls.append(root.left.val)
        dfs(root.left, ls)
        dfs(root.right, ls)
    ls = []  
    dfs(root, ls)
    return ls 

# 杀掉进程 dfs
def killProcess(pid, ppid, kill):  # 有点类似有向无环图的所有可能路径, 父子间关系可连接遍历
    f_c = dict() # 建一个父子dict
    for i in range(len(pid)):
        if ppid[i]:
            f_c[ppid[i]] = f_c.get(ppid[i], []) + [pid[i]]
    res = []
    def dfs(index): # dfs根据f_c字典,遍历完所有可能得子节点
        res.append(index)
        if index in f_c:
            for chird in f_c[index]:
                dfs(chird)
    dfs(kill)
    return res 
pid, ppid, kill = [1,3,10,5], [3,0,5,3], 5
print('需要kill掉的进程: ', killProcess(pid, ppid, kill))

# 二叉树中所有距离为k的节点 距target k距离的节点list
def distanceK(root, target):
    # 还是和有向无环图类似, dfs把树转为图, 然后按照连接关系遍历
    from collections import defaultdict
    graph = defaultdict(set)
    def dfs(root):
        if root.left:
            graph[root.val].add(root.left.val)
            graph[root.left.val].add(root.val)
            dfs(root.left)
        if root.right:
            graph[root.val].add(root.right.val)
            graph[root.right.val].add(root.val)
            dfs(root.right)
    # 把树转为图了~
    dfs(root)
    # 遍历graph
    cur = [target.val]
    visited = {target.val}  
    while k:
        next_time = []
        while cur:
            tmp = cur.pop()
            for node in graph[tmp]: # 当前tmp节点可连接到的所有nodes, 写入next_time
                if node not in visited:
                    visited.add(node)
                    next_time.append(node)
        # 只有跳出cur循环才算遍历到底, 才可以计数1次 
        k -= 1  # 执行k次, 则最后cur中留下的就是与target距离k的节点
        cur = next_time
    return cur 

# 打开转盘锁 dfs 数组 哈希表
def openLock(deadends, target):
    from collections import deque
    # 求'0000'到target序列的最短距离
    deadends = set(deadends)
    if target in deadends or '0000' in deadends: return -1
    q = deque([('0000', 0)])     # 可首尾插入的队列
    vis = {'0000'}
    while q:
        cur, step = q.pop()
        if cur == target:
            return step 
        for i, char in enumerate(cur):
            for dir_ in [-1,1]:  #前后翻转数值变化+-1
                nxt = cur[:i]+str((int(char)+dir_)%10)+cur[i+1:]
                if not nxt in deadends and not nxt in vis:
                    vis.add(nxt)
                    q.appendleft((nxt, step+1))
    return -1 
deadends, target = ["0201","0101","0102","1212","2002"], "0202"
print('打开转盘锁: ', openLock(deadends, target))

# 