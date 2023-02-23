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
print('精度保留5位开方/开根号/求平方根')
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
print('sqart_(5): {}'.format(sqart_(8192)))
# 二分
def mySqrt_half(x):
    if x <= 1:
        return x
    l, r = 0, x
    while True:
        mid = round((r+l)/2)
        if mid**2 <= x < (mid+1)**2:
            break
        elif mid**2 < x:
            l=mid
        else:
            r=mid
    return mid
# 牛顿法 
def mySqrt_niu(x):
    if x <= 1:
        return x
    x0 = x
    while(x0**2 - x) / (2*x0) > 1e-6:
        x0 = x0 - (x0**2 - x) / (2*x0)
    return x0
print('二分求平方根: ', mySqrt_half(8192))
print('牛顿求平方根: ', mySqrt_niu(8192))

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
    im = cv2.imread('/Users/chenjia/Downloads/Smartmore/2022/daydayup/图像马赛克_方块_毛边/lena.png', 0)
    im = add_gauss(im)
    cv2.imwrite('./gauss_lena.jpg', im)
    r = 5
    H, W = im.shape[:2]
    res = np.zeros((H,W))
    for i in range(H-r):
      for j in range(W-r):
        im_ = im[i: i+r, j:j+r]
        res[i: i+r, j:j+r] =  im_*gauss_kernel(r, sigma=0.5)  
    cv2.imwrite('./gauss_filter_lena.jpg', res)
# run_gauss()

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

# 最长公共子序列长度 序列可不连续, 但需保持原单词中的前后相对位置
def longestCommonSubsequence(text1, text2):
    l1, l2 = len(text1), len(text2)
    # dp[i][j]: 记录的是ij之前的元素相同情况, 所以是在比较text1[i-1]和text2[j-1]是否相似
    dp = [[0]*(l2+1) for _ in range(l1+1)]
    for i in range(1, l1+1):
        for j in range(1, l2+1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
t1, t2 = "abcde", "ace"
print('公共子序列最长值:', longestCommonSubsequence(t1, t2))

# 最短公共超序列
def shortestCommonSupersequence(str1, str2):
    l1,l2 = len(str1), len(str2)
    if l1<l2:
        str1, str2 = str2, str1 
        l1,l2 = l2,l1
    # 先找最长公共串, 再把没有的加上
    dp2 = ['']*(l2+1)
    dp = ['']*(l2+1)
    for ch in str1:
        for i in range(l2):
            if str2[i] == ch:
                dp2[i+1]=dp[i]+ch 
            else:
                dp2[i+1] = dp2[i] if len(dp2[i]) > len(dp[i + 1]) else dp[i + 1]
        dp, dp2 = dp2, dp
    lcs = dp[-1]
    if not lcs:  # ==0
        return str1+str2
    res = ''
    i,j,k = 0,0,0
    while k < len(lcs):
        while str1[i] != lcs[k]:
            res += str1[i]
            i += 1
        while str2[j] != lcs[k]:
            res += str2[j]
            j += 1
        res += lcs[k]
        k += 1
        i+=1
        j+=1
    res += str1[i:]+str2[j:]
    return res 
str1, str2 = "abac", "cab"
print('最短的公共超序列, 含s12所有字符且顺序一致: ', shortestCommonSupersequence(str1, str2))

############################################################
# 有序矩阵第k小  行列都有序
def kthSmallest(matrix, k):
    m, n = len(matrix), len(matrix[0])
    lo,hi = matrix[0][0], matrix[-1][-1]  
    while lo<=hi:
        mid = (lo+hi)//2  # 数值上做二分 
        i,j=m-1, 0   # 左下角开始找  
        count = 0
        while i>=0 and j<n:
            if matrix[i][j]<=mid:
                count += (i+1)  # 这一列上面所有的数都是比mid小
                j += 1  # 右移
            else:
                i -= 1
        if count < k:   # mid选小了
            lo = mid+1
        else:
            hi = mid-1
    return lo
matrix, k = [[-5]], 1   # [[1,5,9],[10,11,13],[12,13,15]], 8
print('有序矩阵第k小: ', kthSmallest(matrix, k))


# 数组 除自身外的数组乘积 
# 左右遍历思想 维护left right 值 
def productExceptSelf(nums):
    left,right = 1,1
    l = len(nums)
    res = [0]*l
    for i in range(l):
        res[i] = left   # 现在的res存放的是: 每个位置上元素，其左边需要乘的值
        left *= nums[i]    # left init=1  第一个值的左边乘1啊 没问题的
    for j in range(l-1,-1,-1):
        res[j] *= right   # 这里res开始依次把每个位置上元素，其右边需要乘的值乘上
        # 所以是*=right    right init=1，最后一个值右边可不就是乘1嘛
        right *= nums[j]
    return res
print('除自身外的数组乘积: ', productExceptSelf([1,2,3,4]))

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

# 连续子数组最大和 最大子数组和 
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

# 哈希表 数组
# 外星文是否有序  https://leetcode.cn/problems/lwyVBB/
def isAlienSorted(words, order):
    old = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sentence = ','.join(words)
    # 依次遍历order的各个字符, 在sentence中替换为正常的26字母顺序
    # 完成后看看这个sentence是否符合正常字母的顺序 sorted一致则true
    for i in range(26):
        sentence = sentence.replace(order[i],old[i])
    NewWords = sentence.split(',')
    return NewWords == sorted(NewWords)
words =["word","world","row"]
order = "worldabcefghijkmnpqstuvxyz"
print('外星文是否有序: ', isAlienSorted(words, order))

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

# 翻转1次, 0变1使得连续1的长度最长
def findMaxConsecutiveOnes(nums):
    res, pre_zero, cnt= 0,0,0
    for num in nums:
        cnt += 1
        if num == 0:
            pre_zero = cnt
            cnt = 0
        res = max(res, pre_zero+cnt)
    return res 
nums = [1,0,1,1,0,1] # [1,0,1,1,0]
print('翻转一次0变1使得连续1的个数最大: ', findMaxConsecutiveOnes(nums))

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

# 值和下标之差都在给定的范围内 abs(nums[i] - nums[j]) <= t, abs(i - j) <= k
def containsNearbyAlmostDuplicate(nums, k, t):
    # 桶排序
    bucket = {}
    for i in range(len(nums)):
        b = nums[i]//(t+1) # b是桶的index
        if b in bucket: return True 
        if b-1 in bucket and abs(bucket[b-1]-nums[i])<=t: return True
        if b+1 in bucket and abs(bucket[b+1]-nums[i])<=t: return True
        bucket[b] = nums[i]
        if i >= k:  # i>=k开始才可能满足ij相差k
            bucket.pop(nums[i-k]//(t+1))
    return False 
nums,k,t = [1,5,9,1,5,9], 2,3
print('桶排序思路做: ', containsNearbyAlmostDuplicate(nums, k, t))

# 摇摆序列 摆动序列 数组
# 数组内元素, 增减增减依次
def wiggleMaxLength(nums):
    n = len(nums)
    if n <= 1:
        return n 
    pre_ord = -1 # 默认是先升后降, so初始化的ord为-1先
    res = 1
    for i in range(1, n):
        if nums[i] == nums[i-1]:
            continue
        cur_ord = 1 if nums[i] > nums[i-1] else 0 
        if cur_ord != pre_ord: # 出现+1-1摆动
            res += 1
        pre_ord = cur_ord  # 更新ord状态 
    return res
nums = [1,2,3,4,5,6,7,8,9]
print('摆动数组, 摆动序列: ', wiggleMaxLength(nums))

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
    
# 和至少为k的最短子数组  和>=k  和>=target
# 不可用滑动窗, 因为数组内可能有负数, 则简单左右滑动就不对了...
def shortestSubarray(nums, k):
    from collections import deque
    nums = [0]+nums
    ans = 2**32-1
    s, sum_ = [0],0  # s: 数组前缀和 
    for i in range(1, len(nums)):
        sum_ += nums[i]
        s.append(sum_)
    dq = deque()  # 递增最小堆
    for i, v in enumerate(s):
        while dq and s[dq[-1]] >= v:
            dq.pop()  # 遇到了更小的元素,则丢弃堆的最尾
        while dq and v-s[dq[0]] >= k:  
            # dq内积累到的和为: s[dq[-1]]-s[dp[0]]
            ans = min(ans, i-dq.popleft())  # .popleft即弹出dq[0]
        dq.append(i)
    return -1 if ans==2**32-1 else ans 
nums, k = [2,-1,2], 3
print('和至少为k的子数组长度, 数组可能有负值, 不可滑动窗实现: ', shortestSubarray(nums, k))

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

# 最长连续/上升子序列 (数值上连续上升, 元素的index随意)  最长递增子序列
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
nums = [100,4,200,1,3,2]
print('数组中数值连续, index无要求的, 最长连续值: ', longestConsecutive(nums))

def lengthOfLIS(nums):
    if not nums:
        return 0
    l = len(nums)
    dp = [1]*l
    res = 1
    for i in range(l):
        for j in range(i):  # 注意是i
            if dp[j]+1>dp[i] and nums[i]>nums[j]:  # 状态转移方程
                dp[i] = dp[j]+1
        res = max(res, dp[i])   # 每个index-i更新一个res最大值
    return res
nums = [10,9,2,5,3,7,101,18]
# [2,3,7,101]
print('最长递增数组的长度, index不连续, 数值递增 p就好: ', lengthOfLIS(nums))
# 最长递增数组的 个数 
def findNumberOfLIS(nums):
    l = len(nums)
    dp = [1]*l
    con = [1]*l
    maxlen = 1
    res = 0
    for i in range(l):
        for j in range(i):
            if nums[i] > nums[j]:
                if dp[j]+1 == dp[i]:
                    con[i] += con[j]
                if dp[j]+1>dp[i]:  # 说明找到了更长的最长串 con得重置了
                    dp[i] = dp[j]+1
                    con[i]=con[j]
        maxlen = maxlen if maxlen > dp[i] else dp[i] # 每个i-index更新
    for i in range(l):
        if dp[i] == maxlen:
            res += con[i]
    return res
nums = [2,2,2,2,2] # [1,3,5,4,7]
print('最长递增数组, index不连续的, 个数: ', findNumberOfLIS(nums))

# 要求增序数组且index连续
a = [10, 80, 6, 3, 4, 7, 1, 5, 11, 2, 12, 30, 31]
concoll = [0]
for i in range(len(a) - 1):
    if a[i] < a[i+1]:
        count = concoll[-1] + 1
        concoll.append(count)
    else:
        concoll.append(1)
print('递增数组且index连续, 的长度: ', max(concoll))

# dp  最长字符串链
# 不改变各个word内的字符相对顺序, 每个word可加1个字符, 看看是否可得到后面的word
def longestStrChain(words):
    # 不改变字符相对顺序, 只增加1个字符使word相等
    dp = {}
    for word in sorted(words, key=len): # 按照长度递增顺序排序
        # 依次删除word中的各位字符, 在dp中做累加计数
        dp[word] = max(dp.get(word[:i]+word[i+1:], 0)+1 for i in range(len(word)))
    return max(dp.values())
words = ["abcd","dbqca"] #  ["xbc","pcxbcf","xb","cxbc","pcxbc"]
print('最长字符串链, 前面word加一个字符得到后面的word: ', longestStrChain(words))

# 


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

# 贪心 哈希表
# 形成字符串的最短路径
def shortestWay(source, target):
    i, cnt = 0, 0
    ll = len(target)
    while i < ll:
        tmp = i 
        for s_ in source:
            if i < ll and s_ == target[i]: # 注意处理target的边界
                i += 1
        if tmp!=i:  
            cnt += 1
        else:   # i没变化,即source走完一遍i都没动, t有s没有的字符
            return -1
    return cnt 
print('形成target的最短source路径: ', shortestWay('xyz', 'xzyxz'))

# 贪心算法, 排序   
# 最长数对链 
def findLongestChain(pairs):
    # 按照右边界排序
    pairs = sorted(pairs, key=lambda x: x[1], reverse=False)
    res = 1
    tmp = pairs[0][1]
    for i in range(1, len(pairs)):
        if pairs[i][0] > tmp:  # 后面的左边界大于前面的右边界
            res += 1
            tmp = pairs[i][1]
    return res 
pairs = [[1,2],[7,8],[4,5]]
print('数对链, 后面大于前面, ', findLongestChain(pairs))

# 最大整除子集  输出子数组集 nums[i]%nums[j]==0 or nums[j]%nums[i]==0
def largestDivisibleSubset(nums):
    if not nums or len(nums)==1:
        return nums
    lens = len(nums)
    nums.sort()  # 排序
    dp = [[i] for i in nums]  # 初始化每个子数组包含各个num单个元素
    for i in range(1, lens):
        for j in range(i-1, -1, -1):  # j~i
            if nums[i]%nums[j]==0:
                dp[i] = max(dp[j]+[nums[i]], dp[i], key=len)
    return max(dp,key=len)
nums = [1,2,4,8]
print('最大整除子集: ', largestDivisibleSubset(nums))

# 用最少数量的箭引爆气球
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

# 戳气球 可获得硬币
'''
输入: nums = [3,1,5,8]
输出: 167
解释: 
nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
'''
def maxCoins(nums):
    # dp[i][j]: (i,j)开区间内的最大值, k是(i,j)内被戳破的最后球
    # 最后戳k, 则可获得的值: nums[i]*nums[k]*nums[j], ikj相连,中间无球了都已破
    # i, j代表左右边界, k代表(i,j)内戳球顺序可带来的最大值
    n = len(nums)
    dp = [[0]*(n+2) for _ in range(n+2)]
    nums = [1]+nums+[1]
    # 0~n-1范围是 区间起点
    for i in range(n-1, -1, -1):
        # i+2是循环起点, 因为j表示右边界, 得在i的基础上+2
        for j in range(i+2, n+2):
            # k是ij中间被戳的球, so循环起点是i+1,终点是j
            for k in range(i+1, j):
                # dp[i][k]+dp[k][j]分别是k两边的最大值, 然后附加k戳破能得到的最大值
                dp[i][j] = max(dp[i][j], dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j])
    # 返回(0~n+1)内的最大值  0,n+1都是开区间无法被真的戳破.
    return dp[0][n+1]  # (0,n+1位置都是后面补上去的1啊~)
print('戳气球', maxCoins([3,1,5,8]))

# 删除回文子数组 直到数组空
def minimumMoves(arr):
    n = len(arr)
    f = [[n] * n for i in range(n)]
    g = [None] * n 
    for i, x in enumerate(arr):
        f[i][i] = 1
        # 预处理找到 [i, n) 范围内所有与 a[i] 相等的值的下标，减少无效遍历
        g[i] = [j for j in range(i, n) if x == arr[j]]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if i == j - 1:
                f[i][j] = int(arr[i] != arr[j]) + 1
            else:
                if arr[i] == arr[j]:
                    f[i][j] = f[i + 1][j - 1]
                for k in g[i]:
                    if k >= j:
                        break
                    f[i][j] = min(f[i][j], f[i][k] + f[k + 1][j])
    return f[0][n - 1]

# 合并区间
def merge(intervals):
    if len(intervals) == 1:   
        return intervals     
    begs = []
    ends = []
    for bi in intervals:
        begs.append(bi[0])
        ends.append(bi[1])
    begs.sort()
    ends.sort()  
    l = len(begs)
    tmp1 = 0
    res = []
    for i in range(l-1):
        if begs[i+1] > ends[i]:   
            res.append([begs[tmp1], ends[i]])
            tmp1 = i+1   
        if i == l-2:
            res.append([begs[tmp1], ends[-1]])
    return res
intervals = [[1,3],[2,6],[8,10],[15,18]]
print('合并区间: ', merge(intervals))

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

# 会议室  需要多少个会议室
def minMeetingRooms(intervals):
    mmax = 0
    m0, m1 = max([a[0] for a in intervals]), max([a[1] for a in intervals])
    mmax = max(m0, m1)
    diff = [0]*(mmax+1)
    # 上下车, 起点+1, 终点-1
    for inter in intervals:
        diff[inter[0]] += 1
        diff[inter[1]] -= 1
    ans, cnt = 0,0
    for i in range(mmax+1):
        cnt += diff[i]
        ans = max(ans, cnt)
    return ans 
intervals = [[0,30],[5,10],[15,20]]
print('至少安排多少个会议室: ', minMeetingRooms(intervals))

# 最小时差  : 数组中任意两时间差最小, 小时:分钟 "HH:MM"
def findMinDifference(timePoints):
    # 小时:分钟 "HH:MM" 24*60种不同的时间
    if len(timePoints) > 24*60:
        return 0  # 一定存在重复时间
    # 时:分制转为分钟制度 再升序排序
    min_times = sorted([int(t[:2])*60+int(t[3:]) for t in timePoints])
    min_times.append(min_times[0]+24*60)  # 多加一个数方便处理[i]-[i-1]
    return min([min_times[i]-min_times[i-1] for i in range(1, len(min_times))])
timePoints = ["23:59","00:00"]
print('最小时差: ', findMinDifference(timePoints))

# 两个行程编码数组的积  双指针
# [[1,3],[2,1],[3,2]]: [1,1,1,2,3,3]  每个子数组[value,count]
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

# 字符串list中区分开所有变位词组
def groupAnagrams(strs):
    lens = len(strs)
    if lens <= 1:
        return [strs]
    res_dict = dict()
    for str_ in strs:
        str_map = ''.join(sorted(list(str_)))  # 用了字符串排序. 
        # 不用[0]*256是因为:数组无法作为dict的key,''.join把数组转为string, 仍会使一些不同str_对应同样的key
        if str_map not in res_dict:
            res_dict[str_map] = []
        res_dict[str_map].append(str_)
    res = []
    for k, v in res_dict.items():
        res.append(v)
    return res
strs = ["ddddddddddg","dgggggggggg"] # ["eat", "tea", "tan", "ate", "nat", "bat"]
print('字符串list中所有变位词小组: ', groupAnagrams(strs))

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

# 最长重复连续子串  非暴力方法
# 'aabcaabdaab': 'aab'
def search(L,n,S):
    seen = set()
    for start in range(0, n - L + 1):
        tmp = S[start:start + L]
        if tmp in seen:
            return start
        seen.add(tmp)
    return -1
def longestRepeatingSubstring(S):
    n = len(S)
    left, right = 1, n
    while left <= right:
        mid = left + (right - left) // 2
        if search(mid, n, S) != -1:
            left = mid + 1
        else:
            right = mid - 1
    return left - 1
print('最长重复连续子串长度: ', longestRepeatingSubstring('aabcaabdaab'))   

# 滑动窗: 一边不断r+1扩大窗, 一边遇到重复元素则不断右移左边界缩窗
# 连续子串  无重复最长子串
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

# 滑动窗口 最大值, 返回list 每个滑动范围内的最值依次输出.
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
print('一维maxpooling, 滑动窗内的最大值: ', maxSlidingWindow(nums, k))

# 编辑距离
def minDistance(self, word1, word2):
        # 增 删 换
        m,n = len(word1),len(word2)
        dp = [[0]*(n+1) for i in range(m+1)]
        dp[0] = [i for i in range(n+1)]
        for i in range(m+1):
            dp[i][0] = i
        for i in range(1, m+1):
            for j in range(1, n+1):
                if word1[i-1] == word2[j-1]: # i-1,j-1一致
                    dp[i][j] = dp[i-1][j-1]  # 则ij位置不用处理
                else:  # i-1,j-1不一致, 则ij处增删改选一种 
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
        return dp[-1][-1]

# 字符串, 回文串
# 回文子串个数  每个回文串都得是连续子字符串
def countSubstrings(s):
    ll = len(s)
    count = 0
    dp = [[0]*ll for i in range(ll)]
    for i in range(ll-1, -1, -1):   # 这里注意i的取值是逆向的
    # 因为后面要用dp[i+1][j-1]
        for j in range(i, ll):
            if s[i] == s[j] and (j-i <= 2 or dp[i+1][j-1]):
                dp[i][j] = 1
                count += 1
    return count
print('回文子串个数: ', countSubstrings('aaa'))

# 分割回文串
'''
aab -> [['a','a','b'], ['aa','b']]
'''
def partition(s):
    ll = len(s)
    if not ll:
        return [[]]
    if ll==1:
        return [[s]]
    tmp = []
    for i in range(1, ll+1):
        left = s[:i]
        right = s[i:]
        if left == left[::-1]:  # 保证left部分已经回文
            right = partition(right)
            for j in range(len(right)):
                tmp.append([left] + right[j])
    return tmp
print('把字符串拆分为可能的各个子回文数组, ', partition('aab'))

# 分割回文串 需要的切割次数  最少回文分割
'''
s = "aab" 1次-> 'aa','b', 'a'0次, 'ab'1次->'a','b'
'''
def minCut(s):
    n = len(s)
    g = [[True] * n for _ in range(n)]  # 记录g[i][j]内的回文情况, 0or1
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            g[i][j] = (s[i] == s[j]) and g[i + 1][j - 1]
    f = [float("inf")] * n
    for i in range(n):
        if g[0][i]:
            f[i] = 0
        else:
            for j in range(i):
                if g[j+1][i]:
                    f[i] = min(f[i], f[j]+1)
    return f[-1]
print('分割回文串需要的次数: ', minCut('ab'))

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
    l = len(s)
    if not l:
        return s
    tmp = 0
    res = ''
    dp = [[0]*l for i in range(l)]
    for j in range(l):
        for i in range(j, -1, -1):   # i: 0~j, j: 0~lens
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

# 好的 最长回文串: 个数要是偶数, 且除中心两个元素,其他元素不能相等
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[[0, '*'] for j in range(n)] for i in range(n)]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j] and s[i] != dp[i + 1][j - 1][1]:
                dp[i][j][0] = dp[i + 1][j - 1][0] + 2
                dp[i][j][1] = s[i]
            else:
                dp[i][j] = dp[i][j - 1] if dp[i][j - 1][0] > dp[i + 1][j][0] else dp[i + 1][j]
    return dp[0][-1][0]
print('好的回文串: ', longestPalindromeSubseq("dcbccacdb"))

# 最多删除一个字符, 实现最长回文
def longestPalindromeSubseq(s):
    lens = len(s)
    dp = [[0]*lens for _ in range(lens)]
    for i in range(lens):
        dp[i][i] = 1
    for i in range(lens-1, -1, -1):
        for j in range(i+1, lens):
            if s[i]==s[j]:
                dp[i][j] = dp[i+1][j-1]+2  # ij不相等
            else:  # ij位置不相等, 则可能需要删除, 在i+1j,ij-1中选max
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return dp[0][-1]  # 起点index到终点index的最大值
print('最多可删除一个字符, 实现最长回文: ', longestPalindromeSubseq('bbbab'))

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

# 最多删除k个, 得到回文
def isValidPalindrome(s):
    # lens - 算出的最长回文
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if s[i - 1] == s[n - j]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return k >= n - dp[n][n]
print('删除k个是否可得到回文: ', isValidPalindrome('abbababa'))

# 把球移出界的可能方式  每次上下左右四种走法
def findPaths(m, n, N, i, j):  # m行n列,共可移N步,起点在(i,j)
    dp = [{} for _ in range(N + 1)]   # 移动步数的dp, 字典存的是,当前位置和对应的可移除count
    dp[0][(i, j)] = 1
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]  
    ans = 0
    for step in range(1, N + 1):
        for r, c in dp[step - 1]:  # 这里注意理解下, 有不同的(r,c)key,对应不同的count
            count = dp[step - 1][(r, c)]
            for dr, dc in dirs:
                nr, nc = dr + r, dc + c
                if nr >= m or nc >= n or nr < 0 or nc < 0:
                    ans += count
                    ans %= (10 ** 9 + 7)   #值太大做的一个取模处理
                elif (nr, nc) in dp[step]:
                    dp[step][(nr, nc)] += count
                else:
                    dp[step][(nr, nc)] = count
    return ans
m,n,N,i,j = 1,3,3,0,1# 2,2,2,0,0
print('球移出界的可能情况: ', findPaths(m,n,N,i,j))

# 复制
'''
输入: 3
输出: 3
最初, 只有一个字符 'A'
第 1 步, 使用 Copy All 操作
第 2 步, 使用 Paste 操作来获得 'AA'
第 3 步, 使用 Paste 操作来获得 'AAA'
'''
def minSteps(n):
    # 分解为m个数字相乘等于n, m个数字的和最小 
    res = 0
    for i in range(2, n+1):
        while n%i == 0:
            res += i 
            n //= i 
    return res 

# 轰炸敌人  只有一个炸弹, 只能击杀在同一行同一列没被墙隔开的人 
# W是墙 E是敌人 0是空位可放炸弹   (暴力搜索)
def maxKilledEnemies(grid):
    m, n = len(grid), len(grid[0])
    dirs = [[-1,0],[1,0],[0,1],[0,-1]]  # 上下左右 
    # @cache
    def back(x,y,b):
        if x<0 or x>=m or y<0 or y>=n or grid[x][y] == 'W': return 0
        return (grid[x][y] == 'E')+back(dirs[b][0]+x,dirs[b][1]+y,b)
    return max([sum(back(i,j,x) for x in range(4)) for i in range(m) for j in range(n) if grid[i][j] == '0']+[0])
grid = [["W","W","W"],["0","0","0"],["E","E","E"]]  #  [["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]
print('轰炸敌人: ', maxKilledEnemies(grid))

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

# 最短的包含t中所有字符的s的子串, 得保留t中的字符顺序
def minWindow(s,t):
    if len(s) == len(t) and s == t:
        return s
    start, end = 0, len(s)
    s_index = 0
    t_index = 0
    while s_index < len(s):
        if s[s_index] == t[t_index]:
            t_index += 1
        if t_index == len(t):
            right = s_index
            t_index -= 1
            while t_index >= 0:
                if s[s_index] == t[t_index]:
                    t_index -= 1
                s_index -= 1
            s_index += 1
            if right - s_index < end - start:
                start = s_index
                end = right
            t_index = 0
        s_index += 1
    if end - start == len(s):
        return ""
    else:
        return s[start: end + 1]
print('最短的包含t中所有字符的s的子串, 得保留t中的字符顺序', minWindow("abcdebdde", "bde"))



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

# 某数字二进制表示中, 有多少位1
def fun(n):
    count = 0
    if n < 0:     
        n = n & 0xffffffff
    while n:
        n = n&(n-1)  # n和n-1与,消除n的最尾1
        count += 1
    return count
print('二进制表示有多少个1: ', fun(10))

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

# 乘积最大的连续子数组  数组有正有负 
def maxProduct(nums):
    # 负数偶数个, 则整个数组乘起来就是最值
    # 负数为奇数个, 
        # 1. 则从左边开始乘到最后一个负数为止, 有最值 
        # 2. 同理从右边开始乘到最后一个负数也有一个最值, 
    # so 比较这俩最值即可, 左右分别遍历一次, 维护max值即可
    l = len(nums)
    mmax = nums[0]
    a = 1
    for i in range(l):
        a *= nums[i]
        mmax = a if a > mmax else mmax
        if nums[i] == 0:
            a = 1   # 遇到0重新乘
    a = 1
    for j in range(l-1,-1,-1):
        a *= nums[j]
        mmax = mmax if mmax > a else a
        if nums[j] == 0:
            a = 1
    return mmax
nums = [-2,0,-1] # [2,3,-2,4]
print('数组有正有负, 连续子数组的最大乘积: ', maxProduct(nums))

# 数组有正有负 只把一个元素变成平方值 返回最大子数组和
def maxSumAfterOperation(nums):
    n = len(nums)
    dp0 = nums[0]  # 没替换
    dp1 = nums[0]*nums[0] # 替换了
    res= 0
    if n == 1:
        return max(dp0, dp1)
    for i in range(1, n):
        dp1 = max(dp1+nums[i], dp0+nums[i]*nums[i], nums[i]*nums[i])
        dp0 = max(dp0+nums[i], nums[i])
        res = max(res, dp1) 
    return res 
nums = [1,-1,1,1,-1,-1,1] # [2,-1,-4,-3]
print('一次平方处理返回最大子数组和, 有正有负: ', maxSumAfterOperation(nums))

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

# 最长斐波那契数列 长度 序列
def lenLongestFibSubseq(arr):
    res, lens = 0, len(arr)
    dp = [[0] * lens for i in range(lens)]
    for i, v in enumerate(arr):
        lo, hi = 0, i - 1
        while lo < hi:
            if arr[lo] + arr[hi] < v:
                lo += 1
            elif arr[lo] + arr[hi] > v:
                hi -= 1
            else:  # arr[lo] + arr[hi] == v
                if dp[lo][hi]:
                    dp[hi][i] = dp[lo][hi] + 1  # v可以添加在a[lo][hi]构成的序列后
                else:
                    dp[hi][i] = 3 # lo+hi=i
                res = max(dp[hi][i], res)
                lo += 1
                hi -= 1
    return res
nums = [1,3,7,11,12,14,18] # [1,2,3,4,5,6,7,8]
print('最长斐波那契数列长度, ', lenLongestFibSubseq(nums))

# 复杂版爬楼梯:
'''
输入: cost = [1,100,1,1,1,100,1,1,100,1]
输出: 6, 可选i+1 or i+2处去爬, 花费对应费用.

'''
def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0]*(n+1)  # dp: 爬到n位置需要的最小费用, 没算到达当前位置的费用.
    for i in range(2, n+1):
        dp[i] = min(dp[i-2]+cost[i-2], dp[i-1] + cost[i-1])
    return dp[-1]

# 左上角走到右下角, 可以有的多少种可能? 每步可右or下
def uniquePaths(m, n):
    dp = [[0]*n for i in range(m)]
    # 初始化第一行和第一列, 都是1.
    dp[0] = [1]*n  # 初始化第一行  走法都是1
    for j in range(m):
        dp[j][0] = 1  # 初始化第一列  走法都是1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]
m, n = 3,7
print('左上到右下, 向下或向右, 路径可能:', uniquePaths(m, n))

# 有障碍物, 左上到右下 可能得路径 
def uniquePaths_(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0]*n for i in range(m)]
    dp[0][0] = 1  # 这里需要给[0][0]位置1初始化, 后面会用到
    if grid[-1][-1] == 1: # 终点有障碍物, 则无法到达
        return 0 
    for i in range(m):
        for j in range(n):
            if grid[i][j] != 1: # i,j位置可达, 才需要更新此处的dp值
                if i >= 1 and grid[i-1][j] != 1: # grid[i-1][j] != 1保证此处没堵住
                    dp[i][j] += dp[i-1][j] 
                if j >= 1 and grid[i][j-1] != 1:
                    dp[i][j] += dp[i][j-1]  
    return dp[-1][-1]
grid = [[0,0,0],[0,1,0],[0,0,0]]
print('有障碍, 上到右下路径可能:', uniquePaths_(grid))

# 矩形最小路径和 左上到右下, 最小路径和值
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
grid = [[1,2,3],[4,5,6]] # [[1,3,1],[1,5,1],[4,2,1]]
print('左上到右下最短距离和: ', minPathSum(grid)) 

# 三角形中的最小路径和 
def minimumTotal(triangle):
    lens = len(triangle)  # 三角形的行数 
    dp = triangle[-1]  # init位三角的最后一行数值
    for i in range(lens-2, -1, -1):  # 由下往上走
        for j in range(i+1):
            dp[j] = min(dp[j], dp[j+1])+triangle[i][j]
    return dp[0]
print('由下往上走: ', minimumTotal([[-10]]))# [[2],[3,4],[6,5,7],[4,1,8,3]]))

# 左上到右下, 过程中都会损失值(做减法), 求可抵达右下的最小左上值
# 等价问题:  由下往上走求个最小值  
def calculateMinimumHP(dungeon):
    n, m = len(dungeon), len(dungeon[0])
    dp = [[2**32-1]*(m+1) for _ in range(n+1)]
    dp[n][m-1]=dp[n-1][m] = 1  # # 最后一步, 从左边or上边来, 值得至少为1
    for i in range(n-1, -1, -1):   
        for j in range(m-1, -1, -1):
            mmin = min(dp[i][j+1], dp[i+1][j])
            dp[i][j] = max(1, mmin-dungeon[i][j])
    return dp[0][0]

############################################################
# 分割数组的最大值 变相二分 
# 一个数组分成m份, 使得各自数组的和的最大值 最小  min(max(sm1,sm2,sm3)) 
# 尽量均分m个子数组 
def splitArray(nums, m):
    ll = len(nums)
    if ll == m:
        return max(nums)
    lo, hi = max(nums), sum(nums)
    while lo<hi:   # 数值二分 
        mid = (lo+hi)//2   # 用来做各个区间的划分
        tmp, cnt = 0,1
        for num in nums:
            tmp += num 
            if tmp >mid:
                tmp = num
                cnt += 1
        if cnt>m:  # mid选小了, 导致可分出的组数过多
            lo = mid+1
        else:
            hi = mid 
    return lo
nums, m = [1,4,4], 3
print('分割出m分子数组, 各个子数组的和 的最大值 最小, 尽量均分各个子数组: ', splitArray(nums, m))

# 变相二分  吃香蕉
# 求吃香蕉的速度k 
def minEatingSpeed(piles, h):
    left, right = 1, max(piles)
    while left < right:
        speed = (right+left)//2
        if sum([(speed+p-1)//speed for p in piles]) <= h:
            right = speed
        else:
            left = speed + 1
    return left
piles, h = [30,11,23,4,20], 6
print('吃香蕉的速度: ', minEatingSpeed(piles, h))


# 二分, 数组  两有序数组找中位数 二分: O(log(m+n))
def findMedianSortedArrays(nums1, nums2):
    # 短数组在前, 长数组在后 
    if len(nums1) > len(nums2):
        return findMedianSortedArrays(nums2, nums1)
    infinty = 2**40
    m, n = len(nums1), len(nums2) # m<n
    left, right = 0, m
    median1, median2 = 0, 0  # 前部分最大值, 后部分最小值
    while left <= right:
        i = (left + right) // 2  
        j = (m + n + 1) // 2 - i
        # 前部分: nums1[0 .. i-1] 和 nums2[0 .. j-1]
        # 后部分: nums1[i .. m-1] 和 nums2[j .. n-1]
        nums_im1 = (-infinty if i == 0 else nums1[i - 1]) # nums1[i-1]
        nums_i = (infinty if i == m else nums1[i])  # nums1[i]
        nums_jm1 = (-infinty if j == 0 else nums2[j - 1]) # nums2[j-1]
        nums_j = (infinty if j == n else nums2[j]) # nums2[j]
        if nums_im1 <= nums_j:
            median1, median2 = max(nums_im1, nums_jm1), min(nums_i, nums_j)
            left = i + 1
        else:
            right = i - 1
    return (median1 + median2) / 2 if (m + n) % 2 == 0 else median1
nums1, nums2 = [1,2], [3,4]
print('两个正序数组的中位数: ', findMedianSortedArrays(nums1, nums2))

# 有序数组, 将target插入点, 返回插入index
def searchInsert(nums, target):
    l, r = 0, len(nums)-1
    while l<=r:
        mid = (l+r)//2
        if nums[mid] == target:
            return mid 
        elif nums[mid] < target:
            l = mid+1
        else: 
            r = mid-1 
    return l
print('有序数组插入target的index: ', searchInsert([], 1))

# 出现频率最高的k个数   堆 
def topKFrequent(nums, k):
    map_ = dict()
    for num in nums:
        map_[num] = map_.get(num, 0) + 1
    res = []
    map_ = sorted(map_.items(),key=lambda x:x[1],reverse=True)  
    for i, each in enumerate(map_):
        if i < k:
            res.append(each[0])
        else:
            break
    return res
print('出现频率最高的k个数: ', topKFrequent([1,2,3,3,3,6,6,6,6], 2))

# 山峰数组的顶部
def fengIndex(nums):
    l,r = 0, len(nums)-1
    while l<=r:
        mid = (l+r)//2
        if nums[mid]<nums[mid-1]:  # mid选大了，数组已经在递减了
            r = mid
        elif nums[mid]<nums[mid+1]: # mid选小了，数组还在递增
            l = mid
        else:
            return mid
nums = [1,3,5,7,16,9,8,10]
print(fengIndex(nums))
# 峰值 就是该值大于左右相邻的元素即可
class Solution(object):
    def findPeakElement(self, nums):
        l = len(nums)
        l,r = 0,l-1
        while l<r:
            mid = (l+r)//2
            if nums[mid]>nums[mid+1]:
                r = mid
            else:  # nums[mid]<=nums[mid+1]
                l = mid+1
        return l
s = Solution()
res = s.findPeakElement([1,2,1,3,5,6,4])

# 二分 有序数组 排序数组 都出现2次仅一个出现一次 求这个数  log(n)
def singleNonDuplicate(nums):
    l, r = 0, len(nums)-1
    while l<r:
        mid = (l+r)//2
        # mid偶数mid^1=mid+1; 奇数mid^1=mid-1
        if nums[mid] == nums[mid^1]:
            l = mid + 1
        else:
            r = mid 
    return nums[l] 
print('有序数组中仅出现一次的数: ', singleNonDuplicate([1,1,2,3,3])) 

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
# 删除链表的重复节点  
def deleteDuplicates(head):
    if not head:
        return None
    node = head
    while head.next:
        if head.val==head.next.val:
            head.next=head.next.next
        else:
            head=head.next
    return node

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

# 链表是否有环
def hasCycle(head):
    if not head or not head.next or not head.next.next:
        return False
    fast, slow = head.next.next, head.next 
    while fast != slow:
        if fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        else:
            return False  # 跳不出while,fast提前走到链表尾了: 无环.
    return True 

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

# 设计计算 滑动窗内平均值
class MovingAverage:

    def __init__(self, size: int):
        self.d = [] 
        self.n = size
        self.sumd = 0

    def next(self, val: int) -> float:
        self.d.append(val)
        self.sumd += val
        if len(self.d) > self.n:
             # 窗口过大, sumd和d均 剔除最前val
            self.sumd -= self.d[0] 
            self.d.pop(0) 
        return self.sumd / len(self.d)

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

# 两个链表的第一个重合节点  相交链表 
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
    # main code
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

# 每k个一组反转链表  : 不足k的就不处理
def reverseKGroup(head, k):
    dummy_node = ListNode()
    p = dummy_node
    while True:
        stack = []
        count_k = k 
        tmp = head
        while count_k and tmp:
            stack.append(tmp)
            tmp = tmp.next
            count_k -=1
        if count_k:
            p.next = head
            break
        while stack:  # 逆序弹出栈
            p.next = stack.pop()
            p = p.next
        p.next = tmp
        head = tmp 
    return dummy_node.next

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
                p = p.next # 先把p前移一位,再给这个位置赋予刚刚q的值
                swap(p,q)# 将q的值给p  使得p遍历的节点都小于key
            q = q.next
        swap(head,p)  # 这一步别漏了,把key_ind和之前的head互换 然后分两段使两段均有序
        quicksort(head,p)
        quicksort(p.next,end) 
def sortList(phead):
    if not phead:
        return None
    else:
        quicksort(phead,None) # head and end
    return phead

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

# 单词压缩编码
'''
输入:words = ["time", "me", "bell"]
输出:10
解释:一组有效编码为 s = "time#bell#" 和 indices = [0, 2, 5] .
words[0] = "time" ,s 开始于 indices[0] = 0 到下一个 '#' 结束的子字符串,如加粗部分所示 "time#bell#"
words[1] = "me" ,s 开始于 indices[1] = 2 到下一个 '#' 结束的子字符串,如加粗部分所示 "time#bell#"
words[2] = "bell" ,s 开始于 indices[2] = 5 到下一个 '#' 结束的子字符串,如加粗部分所示 "time#bell#"
'''
def minimumLengthEncoding(words):
    # 每个单词倒序,再排序, 比较相邻字符串
    words_ = []
    for word in words:
        words_.append(word[::-1])
    words = sorted(words_)
    res = 0
    lens = len(words)
    for i in range(lens-1):
        ll = len(words[i])
        if words[i] == words[i+1][:ll]:
            continue 
        res += ll+1 
    return res+len(words[-1])+1
words = ['t'] # ["time", "me", "bell"]
print('单词压缩编码: ', minimumLengthEncoding(words))

###############################################################
# 最大的异或
def max_not_or(nums):
    res = 0
    lens = len(nums)
    for i in range(lens-1):
        for j in range(i+1, lens):
            res = res if res > nums[i]^nums[j] else nums[i]^nums[j]
    return res 
nums = [3,10,5,25,2,8] # [14,70,53,83,49,91,36,80,92,51,66,70]
print('暴力最大的异或: ', max_not_or(nums))

# 排序算法
# 计算右侧小于当前元素的个数   归并排序
def countSmaller(nums):
    arr = []
    res = [0]*len(nums)
    for ind, num in enumerate(nums):
        arr.append([ind, num])
    def merge_sort(arr):
        ll = len(arr)
        if ll <= 1:
            return arr
        mid = (ll)//2
        left = merge_sort(arr[:mid])
        right = merge_sort(arr[mid:])
        return merge(left, right)
    def merge(left, right):
        tmp = []
        i,j=0,0
        while i<len(left) or j<len(right):
            if j==len(right) or i<len(left) and left[i][1]<=right[j][1]:
                tmp.append(left[i])
                res[left[i][0]] += j
                i += 1
            else:
                tmp.append(right[j])
                j += 1
        return tmp 
    merge_sort(arr)
    return res 
print('归并排序做, 右侧小于当前值的个数: ', countSmaller([5,2,6,1]))

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

# 大数相减 大数相乘
def BigMultiply(s1, s2):
    s1 = str(s1)
    s2 = str(s2)
    la = len(s1)
    lb = len(s2)
    res = [0]*(la+lb)
    for i in range(la-1,-1,-1):
        for j in range(lb-1,-1,-1):
            tmp = int(s1[i])*int(s2[j])
            res[i+j+1]+=(tmp%10)
            res[i+j]+=(tmp//10)
            if res[i+j+1] >= 10:
                res[i+j] += 1
                res[i+j+1] %= 10
            if res[i+j] >= 10:
                res[i+j-1] += 1
                res[i+j] %= 10
    return int((''.join(str(ch) for ch in res)).lstrip('0'))
print('大数相乘: ', BigMultiply(789634552, 32110))
def BigMinus(a, b):
    a = [int(item) for item in a]
    b = [int(item) for item in b]
    res = ''
    for i in range(len(b)):
        flag_a = len(a)-1-i
        flag_b = len(b)-1-i
        if a[flag_a]>= b[flag_b]:
            res = str(a[flag_a]-b[flag_b])+res
        else:
            res = str(10+a[flag_a]-b[flag_b])+res
            while a[flag_a-1]==0:
                a[flag_a-1]=9
                flag_a -= 1
            a[flag_a-1] -= 1
    for j in range(len(a)-1-i-1,-1,-1):
        res = str(a[j])+res
    zero_flag=0
    for i in range(len(res)):
        if res[i]!='0':
            zero_flag=1
            break
    if zero_flag==0:
        return 0
    return res[i:]
print('大数相减: ', BigMinus('789634552', '32110'))

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
        # 跳出上面的while,说明右边出现小于key的值,把它放到左边去
        nums[left] = nums[right]
        while left < right and nums[left] <= key:
            left += 1
        # 跳出上面的while,说明左边出现大于key的值,把它放到右边去
        nums[right] = nums[left]
    # 跳出循环,即left>=right
    nums[left] = key
    # 现在完成了小于等于key的在左,大于key的在右
    # 那就递归的把左右分别排序好吧
    # left处等于key值
    quick_sort(nums, low, left-1)
    quick_sort(nums, left+1, high)
    return nums
nums = [5,3,3,7,1,8,1,4]
print('快排: ', quick_sort(nums, 0, 7))

# 分治  给字符串设置计算优先级 得到不同的结果 
def diffWaysToCompute(expression):
    res = []
    ops = {'+': lambda x,y:x+y, '-': lambda x,y:x-y, '*': lambda x,y:x*y}
    for ind in range(1, len(expression)-1):
        if expression[ind] in ops:
            for left in diffWaysToCompute(expression[:ind]):
                for right in diffWaysToCompute(expression[ind+1:]):
                    res.append(ops[expression[ind]](left, right))
    if not res:
        res.append(int(expression))
    return res 
expression = "2*3-4*5" # "2-1-1"
print('字符串设置计算优先级, 得到不同结果, 结果可重复: ', diffWaysToCompute(expression))

# 解数独  数字在一行内只出现一次, 数字在一列内只出现一次, 数字在一个九宫格(从0开始stride为3分布的宫格)内只出现一次
def solveSudoku(board):
    def dfs(pos):
        nonlocal valid
        if pos == len(spaces):
            valid = True
            return
        i, j = spaces[pos]
        for digit in range(9):
            if line[i][digit] == column[j][digit] == block[i // 3][j // 3][digit] == False:
                line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
                board[i][j] = str(digit + 1)
                dfs(pos + 1)
                line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = False
            if valid:
                return
    line = [[False] * 9 for _ in range(9)]
    column = [[False] * 9 for _ in range(9)]
    block = [[[False] * 9 for _a in range(3)] for _b in range(3)]
    valid = False
    spaces = list()
    for i in range(9):
        for j in range(9):
            if board[i][j] == ".":
                spaces.append((i, j))
            else:
                digit = int(board[i][j]) - 1
                line[i][digit] = column[j][digit] = block[i // 3][j // 3][digit] = True
    dfs(0)
    return board
board = [["5","3",".",".","7",".",".",".","."],["6",".",".","1","9","5",".",".","."],[".","9","8",".",".",".",".","6","."],["8",".",".",".","6",".",".",".","3"],["4",".",".","8",".","3",".",".","1"],["7",".",".",".","2",".",".",".","6"],[".","6",".",".",".",".","2","8","."],[".",".",".","4","1","9",".",".","5"],[".",".",".",".","8",".",".","7","9"]]
print('解数独: ', solveSudoku(board))

# N皇后 回溯 数组
def solveNQueens(n):
    m = n*2-1  # 左上下,右上下两条对角线, 自己多算了一次so-1
    ans = []
    col = [0] * n
    on_path, diag1, diag2 = [False]*n, [False]*m, [False]*m
    def dfs(r):
        if r == n:
            ans.append(['.' * c + 'Q' + '.' * (n - 1 - c) for c in col])
            return
        for c, on in enumerate(on_path):
            if not on and not diag1[r + c] and not diag2[r - c]:
                col[r] = c
                on_path[c] = diag1[r + c] = diag2[r - c] = True
                dfs(r + 1)
                on_path[c] = diag1[r + c] = diag2[r - c] = False 
    dfs(0)
    return ans
print('N皇后: ', solveNQueens(4))

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

# 矩阵n次幂
def matrixMul(A, B):
    if len(A[0]) == len(B):
        res = [[0] * len(B[0]) for i in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    res[i][j] += A[i][k] * B[k][j]
        return res
def myPow(x, n):
        if n == 0:
            return 1
        if n == 1:
            return x
        half = myPow(x, n//2) 
        rem = myPow(x, n%2)
        if half != 1 or rem != 1:
            return matrixMul(matrixMul(half, half), rem)
        else:
            return matrixMul(half, half)
a = [[1,2],[3,4]] 
print('矩阵的幂: ', myPow(a, 3))  # [[37, 54], [81, 118]]

# 顺时针
def rotate_shun(A):
    A[:] = zip(*A[::-1])
    return A
# 逆时针矩阵旋转
def roate_ni(A):
    return list(zip(*A))[::-1]

M = [[1,2,3],[4,5,6],[7,8,9]] # np.array()
print('顺逆旋转矩阵: ')
print(M)
print(rotate_shun(M))
M = [[1,2,3],[4,5,6],[7,8,9]]
print(roate_ni(M))

# 数组第k大. 维护最小堆, range(k,lens)这部分大的数持续往前放
def topk_(s, k):
    topk = s[:k] # 先initk个数
    # 遍历k~lens内的数, 遇到大于min(topk)的,就往前放
    for i in range(k, len(s)): 
        if s[i] > min(topk):  
            topk.remove(min(topk))
            topk.append(s[i])
    return min(topk)
print('第k大, 维护最小堆做: ', topk_([3, -1, 2, 10, 55], 3))

# conv2d  二维卷积
import numpy as np 
def conv2d(Input,kernel,padding=0,stride=2):
    h_in, w_in = Input.shape[:2]
    k = kernel.shape[0]   # 默认正方形kernel
    h_out, w_out = (h_in+2*padding-k)//stride+1, (w_in+2*padding-k)//stride+1
    res = np.zeros((h_out, w_out))
    for i in range(h_out):
        for j in range(w_out): 
            # 注意别忘了input的ij移动要乘上stride!!!
            sub_input = Input[i*stride:i*stride+k, j*stride:j*stride+k]
            res[i,j] = np.sum(sub_input * kernel)
    return res
Input = np.array([[40,24,135,1],[200,239,238,1],[90,34,94,1], [1,2,3,4]])
kernel = np.array([[0.0,0.6],[0.1,0.3]])
padding, stride = 0, 2
out_put = conv2d(Input, kernel, padding=padding, stride=stride)
print('Conv2d:', out_put)

# 写个矩阵乘法
def matrix_mul(a, b):
    h, w = a.shape[:2]
    res = np.zeros((h, w))
    for i in range(h):
        a_ = a[i]
        for j in range(w):
            b_ = b[:,j].T
            res[i,j] = np.sum(a_*b_)
    return res 
a = np.array([[1,2],[3,4]])
b = np.array([[1,2],[3,4]])
print('np.matmul矩阵乘法: ', np.matmul(a, b))
print('手动实现矩阵乘法: ', matrix_mul(a, b))

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
    stack = [] # 维护一个递减栈,降序. 存的day-index
    lens = len(temperatures)
    res = [0]*lens
    for day, te in enumerate(temperatures):
        if stack: 
            while stack and temperatures[stack[-1]] < te:
                res[stack[-1]] = day - stack[-1]
                stack.pop() # [-1]计算完了pop掉
        stack.append(day)
    return res 
T = [73,74,75,71,69,72,76,73]
print('天气温度, 更高的温度出现在几天后? ', dailyTemperatures(T)) 

############################################################
# 深度优先dfs, 广度优先bfs
# 矩阵中最长的1线段  类似五子棋的那种排布, 斜线也可
def longestLine(M):
        if not M or not M[0]:
            return 0
        m = len(M)
        n = len(M[0])
        dicts = [[1,0],[0,1],[-1,-1],[-1,1]]  # 这四个相邻!!! 
        max_count = 0
        for i in range(m):
            for j in range(n):
                if M[i][j] == 0:
                    continue
                for k in range(4):    
                    count = 0   
                    x,y = i,j
                    while (x>=0 and y>=0 and x<m and y<n and M[x][y] == 1):
                        x += dicts[k][0]
                        y += dicts[k][1]   
                        count += 1
                    max_count = max(max_count, count) # 在search的四个方向内 
        return max_count
M = [[1,1,1,1],[0,1,1,0],[0,0,0,1]]
print('矩阵中最长的1线段: ', longestLine(M))

# 与目标颜色的最短距离
'''
输入:colors = [1,1,2,1,3,2,2,3,3], queries = [[1,3],[2,2],[6,1]]
输出:[3,0,3]
距离索引 1 最近的颜色 3 位于索引 4(距离为 3)
距离索引 2 最近的颜色 2 就是它自己(距离为 0)
距离索引 6 最近的颜色 1 位于索引 3(距离为 3)
'''
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
colors, queries = [1,1,2,1,3,2,2,3,3], [[1,3],[2,2],[6,1]]
print('与目标颜色的最短距离: ', shortestDistanceColor(colors, queries))

# 整数拆分: 拆分为几个正整数的和 使各元素相乘值最大
def integerBreak(n):
    dp = [0]*(n+1)
    dp[1] = 1  # 整数i对应的最大乘积
    for i in range(2, n+1):
        for j in range(i-1, -1, -1):
            dp[i]=max(dp[i], dp[j]*(i-j))
            dp[i]=max(dp[i], j*(i-j))
    return dp[-1]
print('整数拆分, 得到最大乘积: ', integerBreak(2))

# leetcode286 墙与门  
def wallsAndGates(rooms):
    # bfs广度优先   0:门, -1:障碍物 INF:可通行
    move = [[-1,0],[1,0],[0, -1],[0,1]]  # 上下左右走
    m, n = len(rooms), len(rooms[0])
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

# 扫地机器人
def cleanRoom(robot):
    dirs = [-1, 0, 1, 0, -1]
    visited = set()
    def dfs(x, y, d):
        robot.clean()
        visited.add((x, y))
        for i in range(4):
            cur = (i + d) % 4
            nxt_x, nxt_y = x + dirs[cur], y + dirs[cur+1]
            if (nxt_x, nxt_y) not in visited and robot.move():
                dfs(nxt_x, nxt_y, cur)
                robot.turnRight()
                robot.turnRight()
                robot.move()
                robot.turnLeft()
                robot.turnLeft()
            robot.turnRight()
    dfs(robot.row, robot.col, 0)

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

# 因子的组合
'''
8 = 2 x 2 x 2
  = 2 x 4
'''
def getFactors(n):
    dfs_n_list = collections.defaultdict(list)
    def dfs(n, l):
        if n in dfs_n_list:
            ans = []
            for fcmb in dfs_n_list[n]:
                if fcmb[0] >= l:  # 因子list升序排, so 0index>l
                    # 可加入ans
                    ans.append(fcmb)
            return ans
        # n没出现在dfs_n_list是新的因子
        ans = []
        for i in range(l, int(np.sqrt(n)) + 1):
            if n % i == 0:
                ans.append([i, n//i])
                for fcmb in dfs(n//i, i):
                    ans.append([i] + fcmb) # [i]往前放,保持因子list升序
        dfs_n_list[n] = ans
        return ans
    return dfs(n, 2) # 因子>1故从2开始
print('因子的所有组合: ', getFactors(8))

# 矩阵中的最长递增序列  上下左右可相连
def longestIncreasingPath(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    lookup = [[0]*n for _ in range(m)]
    def dfs(i,j):  # 实现ij位置的最大增续值计算
        if lookup[i][j] != 0:  # 已经遍历过
            return lookup[i][j]
        res = 1
        for x, y in [[-1, 0], [1, 0], [0, 1], [0, -1]]: # 四个方向
            tmp_i = x + i
            tmp_j = y + j
            if 0 <= tmp_i < m and 0 <= tmp_j < n and \
                    matrix[tmp_i][tmp_j] > matrix[i][j]:  # 增序要求
                res = max(res, 1 + dfs(tmp_i, tmp_j))
        lookup[i][j] = max(res, lookup[i][j])
        return lookup[i][j]
    return max(dfs(i, j) for i in range(m) for j in range(n))
matrix = [[3,4,5],[3,2,6],[2,2,1]] # [[9,9,4],[6,6,8],[2,1,1]]  
print('矩阵中的最长递增路径: ', longestIncreasingPath(matrix))

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

# 根节点到叶节点的路径数字之和
'''
输入:root = [4,9,0,5,1]
输出:1026
解释:
从根到叶子节点路径 4->9->5 代表数字 495
从根到叶子节点路径 4->9->1 代表数字 491
从根到叶子节点路径 4->0 代表数字 40
数字总和 = 495 + 491 + 40 = 1026
'''
def sumNumbers(root):
    if root is None:
            return 0
    if not any([root.left, root.right]):
        return root.val
    left, right = 0, 0
    if root.left:
        root.left.val += root.val * 10
        left = sumNumbers(root.left)
    if root.right:
        root.right.val += root.val * 10
        right = sumNumbers(root.right)
    return left + right

# 把每个节点的值替换为: 大于ta值的所有value之和
# 逆向中序遍历 所有大于等于节点的值之和
def convertBST(root):
    def dfs(root: TreeNode):
        nonlocal total
        if root:
            dfs(root.right)
            total += root.val
            root.val = total
            dfs(root.left)
    total = 0
    dfs(root)
    return root

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right 

# 二叉搜索树迭代器
class BSTIterator:
    def __init__(self, root: TreeNode):
        self.pos=-1
        self.data=[]
        self.in_order(root)
    def next(self):
        self.pos+=1
        return self.data[self.pos]
    def hasNext(self):
        if self.pos>=len(self.data)-1:
            return False
        return True
    def in_order(self,root):
        if not root:return
        self.in_order(root.left)
        self.data.append(root.val)
        self.in_order(root.right)

# 是否存在 两个节点之和等于target 
def findTarget(root, k):
    in_order = []
    def in_traversal(node):
        if node is None:
            return
        in_traversal(node.left)
        in_order.append(node.val)
        in_traversal(node.right)
    in_traversal(root)
    l, r = 0, len(in_order) - 1
    while l < r:
        tmp = in_order[l] + in_order[r]
        if tmp == k:
            return True
        elif tmp < k:
            l += 1
        else:
            r -= 1
    return False


# 树的独生节点
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

# 二叉树向下的路径 节点之和 等于 target
class Solution:
    ans = 0
    def pathSum(self, root: TreeNode, targetSum: int) -> int:
        # dfs+前缀和
        sum_map = {0:1}
        def fun(node, pre_sum):
            x = node.val + pre_sum
            self.ans += sum_map.get(x-targetSum,0)
            sum_map[x] = sum_map.get(x, 0)+1
            if node.left:
                fun(node.left, pre_sum+node.val)
            if node.right:
                fun(node.right, pre_sum+node.val)
            sum_map[x] -= 1
            if sum_map[x]==0:
                del sum_map[x]
        if not root:
            return 0
        fun(root, 0)
        return self.ans 

# 二叉树中序遍历, 然后结点s变成一个长条链
class Solution:
    def increasingBST(self, root: TreeNode) -> TreeNode:
        # 中序遍历(左根右), 再生成一个链条树 
        self.pre = TreeNode(0)
        res = self.pre
        def fun(root):
            if not root:
                return 
            fun(root.left)
            root.left = None
            self.pre.right = root
            self.pre = self.pre.right
            fun(root.right)
        fun(root)
        return res.right

# 不同的二叉搜索树
def numTrees(n):
    dp = [0] * (n+1)
    dp[:2] = 1,1
    for i in range(2,n+1):
        for j in range(1,i+1):
            dp[i] += dp[j-1] * dp[i-j]
    return dp[n]

# 二叉树的中序后继 (中序遍历, 节点p的后一个节点)
class Solution:
    def inorderSuccessor(root,p):  # 中序遍历: 左根右
        res = None 
        val = p.val
        while root:
            if root.val > val:
                res = root
                root = root.left
            else:
                root = root.right
        return res 

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

# 二叉树最底层最左边的值 
def findBottomLeftValue(root):
    # 层次遍历 
    q = deque([root])
    while q:
        node = q.popleft()  # 弹出左值 
        if node.right: q.append(node.right) # 上一层的右 
        if node.left:  q.append(node.left)  # 下一层的左 
    return node.val

# 二叉树深度  
# 递归 
def TreeDepth(pRoot):
    if pRoot == None:
        return 0
    lDepth = TreeDepth(pRoot.left)
    rDepth = TreeDepth( pRoot.right)
    return max(lDepth, rDepth) + 1
# 非递归: bfs队列, 层次遍历
import collections
class Solution(object):
    def maxDepth(self, root):
        if not root:
            return 0
        que = collections.deque()
        que.append(root)
        high = 0
        while que:
            cur_len = len(que)
            for i in range(cur_len):
                tmp = que.popleft()
                if tmp.left:
                    que.append(tmp.left)
                if tmp.right:
                    que.append(tmp.right)
            high += 1
        return high

# 二叉树的右侧视图
'''
输入: [1,2,3,null,5,null,4]
输出: [1,3,4]
'''
def rightSideView(root):
    d = {}
    def f(root, i):   # i为树的深度
        if root:
            d[i] = root.val
            f(root.left, i+1)
            f(root.right, i+1)  # right要放在后面，因为要把前面的left覆盖掉
    f(root, 0)
    return list(d.values())

# 二叉树剪枝 某子树节点值全是0无1, 则这个子树可剪掉
def pruneTree(root):        
    if not root:
        return root
    root.left = self.pruneTree(root.left)
    root.right = self.pruneTree(root.right)
    # 只剩下root无左右且root.val还等于0 
    if root.val == 0 and not root.left and not root.right:
        return None 
    return root

# 监控二叉树: 摄影头可监视其父对象、自身及其 直接子对象
def minCameraCover(root):
    def dfs(root):
        if not root:
            return [float("inf"), 0, 0]
        # a: root必须放摄像头的情况下, 覆盖整棵树需要的摄像头 
        # b: 覆盖整棵树需要的摄像头数目无论root是否放
        # c: 覆盖两棵子树需要的摄像头数目无论root是否被监控
        la, lb, lc = dfs(root.left)
        ra, rb, rc = dfs(root.right)
        a = lc + rc + 1
        b = min(a, la + rb, ra + lb)
        c = min(a, lb + rb)
        return [a, b, c]
    a, b, c = dfs(root)
    return b

# 二叉树最大路径和  dfs 递归
'''
输入:root = [-10,9,20,null,null,15,7]
输出:42
'''
def maxPathSum(root):
    self.res = -sys.maxsize
    self.dfs(root)
    return self.res
# self.dfs()实现的就是root节点加左右子节点的最大值
# 左右出现负的话, 则dfs return的就是root.val
def dfs(self, root: TreeNode) -> int:
    if not root:
        return 0
    l_max = self.dfs(root.left)
    r_max = self.dfs(root.right)
    temp_sum = root.val
    temp_sum = max(temp_sum, temp_sum+l_max)
    temp_sum = max(temp_sum, temp_sum+r_max)
    # 维护全局变量
    self.res = max(self.res, temp_sum)
    return max(0, l_max, r_max) + root.val

# 寻找二叉树的叶子节点
'''
输入: [1,2,3,4,5] 
          1
         / \
        2   3
       / \     
      4   5    
输出: [[4,5,3],[2],[1]]
'''
def findLeaves(root):
    # 按照子树的高度, 分组的 
    res = []
    def fun(root):
        if not root:
            return 0
        left = fun(root.left)
        right = fun(root.right)
        height = max(left, right)
        if len(res) == height:
            res.append([])
        res[height].append(root.val)
        return height+1
    fun(root)
    return res 

# N叉树的直径  树中两个点的最远距离 dfs
def diameter(root):
    def dfs(node, ans):
        d = 0  # 统计已经走过的树深 
        for node_c in node.children:
            tmp, ans = dfs(node_c, ans)
            ans = max(ans, d+tmp+1)
            d = max(d, tmp+1)
        return d, ans
    ans = 0
    if not root:
        return 0
    d, ans = dfs(root, ans)
    return ans 

# 往完全二叉树添加节点
class CBTInserter:
    '''
    BFS: 用两个队列表示当前层和下一层, 初始化就要不断地往下遍历, 使当前层是未填满的一层
    插入: 判断当前层的第一个元素, 如果没有左子树, 就插入到该节点的左子树, 然后在Q2中更新
    如果没有右子树, 那除了插入操作以外, 还需要把这个节点从Q中pop出来, 然后更新Q2, 如果Q空那就让Q=Q2
    '''
    def __init__(self, root: TreeNode):
        self.root = root
        self.Q  = deque([self.root])  #当前层
        self.Q2 = deque([])  #下一层
        while self.Q:
            a = self.Q[0]
            if not a.left: 
                break
            elif not a.right:
                self.Q2.append(a.left)
                break
            else:
                cur = self.Q.popleft()
                self.Q2.append(cur.left)
                self.Q2.append(cur.right)
                if not self.Q:
                    self.Q = self.Q2
                    self.Q2 = deque([])
    def insert(self, v: int) -> int:
        cur = self.Q[0]
        if not cur.left:
            cur.left = TreeNode(v)
            self.Q2.append(cur.left)
        else:
            cur.right = TreeNode(v)
            self.Q2.append(cur.right)
            self.Q.popleft()
            if not self.Q:
                self.Q = self.Q2
                self.Q2 = deque([])
        return cur.val
    def get_root(self) -> TreeNode:
        return self.root

# 找到二叉树中最近的右侧节点  dfs
# 层次遍历视角, 找到和u同层的, 第一个右边节点 
def findNearestRightNode(root, u):
    self.target = float('inf')
    def dfs(node, i=1):
        if node:
            if node == u:
                self.target = i
            else:
                return node if i > self.target else dfs(node.left, i*2) or dfs(node.right, i*2+1)
    return dfs(root)

# 打家劫舍 树形 有连接的两个节点不可同时偷
def rob(root):
    if not root:
        return 0
    def helper(root):  # return: [包含node不含左右子, 不含node只有左右子]
        if not root:
            return [0,0]
        left = helper(root.left)
        right = helper(root.right)
        # 有根节点root 就不能有root.left root.right
        # 0表示包含当前节点  1表示不包含
        rob = root.val + left[1] + right[1] 
        skip = max(left) + max(right)
        return [rob, skip]  # 0表示包含当前节点  1表示不包含
    res = helper(root)
    return max(res)

# 二叉树的每一层内的最大值   二叉树每层的最大值
def largestValues(root):
    res = []
    q = []
    if not root:
        return res
    q.append(root)
    while q:
        level_len = len(q)  # 本层节点个数 
        mmax = -2**31
        for i in range(level_len):
            node = q.pop(0)  # 越前面层的节点越往前放 
            mmax = mmax if mmax > node.val else node.val  
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        res.append(mmax)
    return res

# 删除给定值的叶子节点    递归, 简单直接 
# 只要节点的值和key/target相等, 就删掉这个节点
def removeLeafNodes(root):
    if not root:
        return None
    root.left = removeLeafNodes(root.left, key)
    root.right = removeLeafNodes(root.right, key)
    # 根节点val就等于target的情况
    if root.val == key and not root.left and not root.right:
        return None 
    return root 

# 打开转盘锁 dfs 数组 哈希表 开密码锁
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

# 网格左上走到右下代价最小值  leetcode1368 
def minCost(grid):  # grad中1234分别代表向右左上下走, 不按1234指示走则const+1
    # bfs: 理解为所有位置均可到达, 但是会叠加代价
    dir_map = {1:[0,1], 2:[0,-1],3:[1,0],4:[-1,0]}
    # 走的方向与原来的方向不一致则cost+1, 一致则不变cost
    m, n = len(grid), len(grid[0])
    const = [[2*32-1]*n for _ in range(m)]  # const init拉到最大值
    const[0][0] = 0
    from collections import deque
    que = deque([(0, 0)]) # 优先可走的往前放, appendleft, popleft
    while que:
        temp_first = que.popleft()  # 弹出最左元素位置
        x, y = temp_first[0], temp_first[1]
        for k, chang_xy in dir_map.items():
            x_, y_ = x+chang_xy[0], y+chang_xy[1]
            if 0<= x_ < m and 0 <= y_ < n:  
                # grid处的值1234和走的方向上下左右一致则无代价,否则cost+1
                cur_cost  = 0 if k == grid[x][y] else 1
                # 从xy走到x_y_需要的代价
                cost2x_y_ = const[x][y] + cur_cost
                if cost2x_y_ < const[x_][y_]: # xy过来x_y_更快 
                    const[x_][y_] = cost2x_y_  # 更新x_y_处的代价值 
                    if cur_cost == 0:  # 如果走到这步没花代价, 则把x_,y_往前放优先. 基于ta再往后走
                        que.appendleft((x_, y_))
                    else:
                        que.append((x_, y_))
    return const[-1][-1]
grid = [[4]]
print('左上走到右下的最小代价, 1234右左上下: ', minCost(grid))

# 是否可二分图 拆分为无相互连接的两个部分图
def isBipartite(graph):
    n = len(graph)
    color = [0]*n  # 0表示未访问, 1表示色1, -1表示色2 
    queue = []
    for i in range(n):
        if color[i]!=0: continue  # 被分组好染好色了则不管
        queue.append(i)  
        color[i] = 1 # 加入色1队列
        while queue:
            cur = queue.pop(0)
            for ner in graph[cur]: # 遍历cur的全部相连节点
                if color[ner] == color[cur]: return False 
                if color[ner] == 0:
                    color[ner] = -color[cur]  # 相邻的得染不同的色
                    queue.append(ner)
    return True
graph = [[1,3],[0,2],[1,3],[0,2]]  # [[1,2,3],[0,2],[0,1,3],[0,2]]
print('是否可二分图: ', isBipartite(graph))

def possibleBipartition(n, dislikes):
    # bfs  染色问题(有dislikes关系的不可染一样的色), 二分图问题 
    g = [[] for _ in range(n+1)]
    # 构建图, 把不喜欢关系(染色连接关系) 放入图内
    for i in range(len(dislikes)):
        g[dislikes[i][0]].append(dislikes[i][1])
        g[dislikes[i][1]].append(dislikes[i][0])
    # 染色了表示分好组了value=1, 没则为0
    colors = [0]*(n+1)
    for i in range(n):
        if colors[i] != 0:  # 已经分好组了,不管
            continue
        colors[i] = 1
        queue = [i]  # 开始找i的图连接关系
        while queue:
            cur = queue.pop()
            for hate in g[cur]:
                if colors[hate] == colors[cur]:
                    return False  # 不能为同样的色
                if colors[hate] == 0: # 还没染色, 那就染成不一样的吧
                    colors[hate] = -colors[cur]
                    queue.append(hate) # 并且加入队列
    return True 
n, dislikes = 5, [[1,2],[2,3],[3,4],[4,5],[1,5]]
print('染色, 二分图, ', possibleBipartition(n, dislikes))

# 连通网络的操作次数 
def makeConnected(n, connections):
    # dfs遍历无向图 
    def dfs(node):
        vis[node] = 1
        for ner in graph[node]:
            if not vis[ner]:
                dfs(ner) 
    if len(connections) < n-1:
        return -1
    graph = [set() for _ in range(n)]
    vis = [0]*n 
    # connections包含的所有可连接信息写入graph
    for conn in connections:
        graph[conn[0]].add(conn[1])
        graph[conn[1]].add(conn[0])
    res = 0  
    for i in range(n):
        if not vis[i]:
            dfs(i)
            res += 1  # vis在dfs中会被置1很多位,没被处理到的就是缺失连接的, 就得res+1
    return res - 1
connections, n = [[0,1],[0,2],[0,3],[1,2],[1,3]], 6
print('无向图遍历, 连通网络的操作次数', makeConnected(n, connections))

# 连通分量: n个节点 互相有一些连接 求连通个数 
def countComponents(n, edges):
    import collections 
    d = collections.defaultdict(set)
    # 建立无向图连接关系 
    for e in edges:
        d[e[0]].add(e[1])
        d[e[1]].add(e[0])
    todo = set(range(n))
    def dfs(x):
        for y in d[x]:
            if y in todo:
                todo.remove(y)
                dfs(y)
    r = 0
    while todo:
        x = todo.pop()
        r += 1
        dfs(x)
    return r
n, edges = 5, [[0,1], [1,2], [2,3], [3,4]] # [[0, 1], [1, 2], [3, 4]]
print('求连通分量: ', countComponents(n, edges))

# 查找集群内的关键连接  leetcode1192
'''
无向图要么环要么链, 有链的话(只有一个连接,), 没入环之前的点都是关键连接  bfs遍历
剩下找两个环之间的唯一通连接, 也是关键连接. 
'''  
def criticalConnections(n, connections):
    if n == 2:#如果只有两点，两点一线必然是关键边
        return connections
    def dfs(x):#深搜，寻找两点之间是否相通
        if x == j:#如果到达终点
            return True
        for y in d.get(x, []):
            if degree[y] > 1 and (x, y) != (i, j) and visit[y] == 0:#不需要走链，且不能起点和终点直连且下个点未到达过
                visit[y] = 1
                if dfs(y):
                    return True
        return False
    res = []
    degree = [0 for i in range(n)]#边数
    d = {}#每个点能到达的其他点列表
    for i, j in connections:
        if i not in d:
            d[i] = set()
        if j not in d:
            d[j] = set()
        d[i].add(j)
        d[j].add(i)
        degree[i] += 1#边+1
        degree[j] += 1#边+1
    deq = collections.deque()#bfs队列
    for i in range(n):
        if degree[i] == 1:#如果为1，则是链头
            deq.append(i)
    while deq:
        i = deq.popleft()
        for j in d[i]:
            if degree[j] > 1:
                res.append([i, j])#加入关键边
                degree[j] -= 1#边数减1
                if degree[j] == 1:#如果链未入环，继续遍历
                    deq.append(j)
    for i, j in connections:
        if (degree[i] == 3 and degree[j] >= 3) or (degree[j] == 3 and degree[i] >= 3):#如果满足上面的两个条件
            visit = [0 for i in range(n)]
            if not dfs(i):#深搜判断是否为关键边
                res.append([i, j])
    return res


# 日程表
'''
输入:
["MyCalendar","book","book","book"]
[[],[10,20],[15,25],[20,30]]
输出: [null,true,false,true]
'''
class MyCalendar:
    def __init__(self):
        self.starts = []
        self.ends   = []
    def book(self, start: int, end: int) -> bool:
        if self.starts == []:
            self.starts.append(start)
            self.ends.append(end)
            return True
        s = bisect.bisect_left(self.starts, start)
        if s == 0 and end <= self.starts[0]:
            self.starts[s:s] = [start]
            self.ends[s:s]   = [end]
            return True
        elif start < self.ends[s-1]:
            return False 
        elif s == len(self.starts) or end <= self.starts[s]:
            self.starts[s:s] = [start]
            self.ends[s:s]   = [end]
            return True
        else:
            return False      

####################################################################
# 回溯
# 含有k个元素的组合   元素范围在1~n, 
'''
输入: n = 4, k = 2
输出:
[
  [2,4],
  [3,4],
  [2,3],
  [1,2],
  [1,3],
  [1,4],
]
'''
def combine(n,k):
    if k==1:
        return [[t] for t in range(1, n+1)]
    if n == k:
        return [[t for t in range(1, n+1)]]
    ans = []
    for t in combine(n-1, k-1):
        t.append(n)
        ans.append(t)
    return ans+combine(n-1, k)
print('含有k个元素的组合: ', combine(4, 2))

# 删除无效的括号
# 回溯  bfs
def removeInvalidParentheses(s):
    # bfs 回溯
    def isValid(s):  # 由前往后)不可以>左括号, 返回左右括号个数相等
        cnt = 0
        for c in s:
            if c == '(': cnt += 1
            elif c == ')': cnt -= 1
            if cnt < 0: return False # 只用中途cnt出现了负值就return
        return cnt == 0
    level = {s}
    while True:
        valid = list(filter(isValid, level))
        if valid: return valid
        # 进入下一层 
        next_level = set()
        for item in level:
            for i in range(len(item)):
                if item[i] in '()':
                    # 剔除了i处的左or右括号
                    next_level.add(item[:i]+item[i+1:])
        level = next_level
s = "(a)())()" # "()())()"
print('删除最少量无效括号, 得到有效字符串: ', removeInvalidParentheses(s))

# 计算AUC值 
def AUC(label, pre):
    pos = [i for i in range(len(label)) if label[i] == 1]
    neg = [i for i in range(len(label)) if label[i] == 0]
    auc = 0
    for i in pos:
        for j in neg:
            if pre[i] > pre[j]:
                auc += 1
            elif pre[i] == pre[j]:
                auc += 0.5
    return float(auc) / (len(pos)*len(neg))
label = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
pre = [0.9, 0.7, 0.6, 0.55, 0.52, 0.4, 0.38, 0.35, 0.31, 0.1]
print(AUC(label, pre))

# 二维01矩阵 连通域个数, 最大连通域值  
# bfs dfs  leetcode200 leetcode463 
# 连通域个数 岛屿个数
def numIslands(grid):  
    def dfs(grid, i, j):
        if (i<0 or j<0 or i >= m or j >= n or grid[i][j] != '1'):
            return  # ij过边界, grid[ij]是'0'水
        # 除去以上, 就是没过界且为'1'陆地
        grid[i][j] = '2'  # 把岛屿染成2避免main代码中重复计算
        dfs(grid, i+1, j)
        dfs(grid, i-1, j)
        dfs(grid, i, j+1)
        dfs(grid, i, j-1)  # 上下左右四邻域 
    res = 0
    m, n = len(grid), len(grid[0])
    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':  # 碰到陆地了, 那就找能连上的所有陆地
                dfs(grid, i, j)
                res += 1
    return res 
grid = [["1","1","0","0","0"],
        ["1","1","0","0","0"],
        ["1","0","0","0","0"],
        ["0","1","0","1","1"]]
print('连通域个数, 岛屿个数: ', numIslands(grid))

# dfs 省份数量 同岛屿问题类似 连通域个数
def findCircleNum(M):
    def dfs(M, vis, i):
        for j in range(len(M)): # 省份间的连接关系是正方形的~
            if M[i][j] == 1 and not vis[j]:
                vis[j] = 1
                dfs(M, vis, j)
    # dfs 
    lens = len(M)
    vis = [0]*lens
    res = 0
    for i in range(lens):
        if not vis[i]:  # 没被访问连接
            dfs(M, vis, i)
            res += 1
    return res 
M = [[1,0,0],[0,1,0],[0,0,1]] # [[1,1,0],[1,1,0],[0,0,1]]
print('省份数量, 类岛屿问题: ', findCircleNum(M))

# 最大连通域  8邻域 
def max_one_area(M):
    ners = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]] # 八邻域
    # 偶尔也会四邻域  ners = [[-1,0],[0,-1],[0,1],[1,0]] 
    m = len(M)
    n = len(M[0])
    mmax = 0
    visited = [[0]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            count = 0
            if M[i][j] and not visited[i][j]:
                count += 1
                visited[i][j] = 1
                queue = []
                queue.append([i,j])
                while queue:
                    cur = queue.pop(0)  
                    for ner in ners:    
                        x = cur[0]+ner[0]
                        y = cur[1]+ner[1]
                        if (x>=0 and y>=0 and x<m and y<n \
                            and M[x][y] and not visited[x][y]):
                            count += 1
                            visited[x][y] = 1   
                            queue.append([x,y])
            mmax = max(mmax, count)
    return mmax
M = [[0,0,0,0],[1,1,0,1],[0,1,1,1],[0,1,0,0],[0,0,0,1]]
print(np.array(M))
print('最大连通域: ', max_one_area(M))

# 矩阵中 距离最近0的距离  矩阵中的距离 
# 矩阵各个index距离最近的0的距离, 结果仍返回一个矩阵
def updateMatrix(mat):
    m, n = len(mat), len(mat[0])
    ners = [[-1,0],[1,0],[0,-1],[0,1]]
    queue = []
    for i in range(m):
        for j in range(n):
            if mat[i][j] == 0:
                queue.append([i,j])
            else:
                mat[i][j] = m+n  # mark位m+n, 后面会被附近的0找
    while queue:
        cur = queue.pop(0)
        for ner in ners:
            x, y = cur[0]+ner[0], cur[1]+ner[1]
            if x>=0 and y>=0 and x<m and y<n and mat[x][y]>mat[cur[0]][cur[1]]+1:  # mat[x][y]>mat[cur[0]][cur[1]]+1表明x_y_位置上是1,需要去找附近的0
                mat[x][y] = mat[cur[0]][cur[1]]+1
                queue.append([x,y])
    return mat
mat = [[0,0,0],[0,1,0],[1,1,1]]
print('矩阵中距离最近0的距离, 返回矩阵res:', updateMatrix(mat))

# 岛屿周长  0为水 1为陆地 
def islandPerimeter(grid):
    from scipy.signal import convolve2d
    # 左上角有有相近就-2, 上or下相邻则+1
    return int(abs(convolve2d(grid,[[-2,1],[1,0]])).sum())
grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
print('岛屿的周长: ', islandPerimeter(grid))

# 课程表
def canFinish(numCourses, prerequisites):
    edges = collections.defaultdict(list)
    visited = [0] * numCourses
    result = list()
    valid = True
    for info in prerequisites:
        edges[info[1]].append(info[0])
    def dfs(u: int):
        nonlocal valid
        visited[u] = 1
        for v in edges[u]:
            if visited[v] == 0:
                dfs(v)
                if not valid:
                    return
            elif visited[v] == 1:
                valid = False
                return
        visited[u] = 2
        result.append(u)
    
    for i in range(numCourses):
        if valid and not visited[i]:
            dfs(i)
    return valid

# 课程表  n门课, 数组记录修课的前后关系, bfs思路做
# 有向图问题  
def findOrder(numCourses, prerequisites):
    edges = collections.defaultdict(list) # value定义为list
    indeg = [0] * numCourses
    result = list()
    for info in prerequisites:
        edges[info[1]].append(info[0])  # 被依赖的情况
        indeg[info[0]] += 1  # 存储各课程依赖其他课程的情况 
    # q内均无依赖 
    q = collections.deque([u for u in range(numCourses) if indeg[u] == 0])
    while q:
        u = q.popleft()  # q内的顺序是从小到大的
        result.append(u)
        for v in edges[u]:
            indeg[v] -= 1
            if indeg[v] == 0:  # indeg[x]==0表示无依赖了, 可入res
                q.append(v)
    if len(result) != numCourses:
        result = list()
    return result
n, prerequisites = 4, [[1,0],[2,0],[3,1],[3,2]]
print('有向图问题, bfs实现, 课程安排: ', findOrder(n, prerequisites))

# 直线上最多的点数
def maxPoints(points):
    from collections import Counter 
    res = 0
    lens = len(points)
    for i in range(lens-1):
        dict_ = Counter() # 每个点都维护一个Counter()
        for j in range(i+1, lens):
            dif_x, dif_y = points[i][0]-points[j][0], points[i][1]-points[j][1]
            if dif_x == 0: 
                dict_[(points[i][0], )] += 1
            else:
                dict_[dif_y/dif_x] += 1
            if dict_:
                res = max(res, max(dict_.values()))
    return res+1
points = [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
print('直线上最多的点数: ', maxPoints(points))

# 表示数值的字符串
def isNumber(s):
    point_done = False
    e_done = False
    if s and s[0] in '+-.':
        s = s[1:]
    if len(s) == 0:
        return False
    for i in range(len(s)):
        if s[i] in 'eE':
            s = s[i + 1 :]
            e_done = True
            break
        elif s[i] == '.':
            s = s[i + 1 :]
            point_done = True
            break
        elif s[i] < '0' or s[i] > '9':   # 这句用elif not s[i].isdigit():也行
            return False
    if point_done == True:
        for i in range(len(s)):
            if s[i] in 'Ee':
                s = s[i + 1 :]
                e_Done = True
                break
            elif s[i] < '0' or s[i] > '9':
                return False
    if e_done == True:
        if len(s) == 0:
            return False
        if s[0]  in '+-':
            s = s[1:]
        if len(s) < 1:
            return False
        for x in s:
            if x < '0' or x > '9':
                return False
    return True
print('表示数值的字符串: ', isNumber('++1'))

#########################################################
# 栈
heights = [2,1,5,6,2,3]
def largestRectangleArea(heights):   # 这个方法更通用!!! 
    heights = [0]+heights+[0]
    stack = [0]
    res = 0
    for i in range(1, len(heights)):
        while stack and heights[i]<heights[stack[-1]]:
            high = heights[stack[-1]]
            stack.pop()
            width = i-stack[-1]-1
            res = max(res, width*high)
        stack.append(i)
    return res 
print('通用: 柱状图最大矩形, 其实就是直方图内的最大矩形: ', largestRectangleArea(heights))

# 后缀表达式  感觉主要考lambda表达式
def evalRPN(tokens):
    f1 = lambda a, b: b+a
    f2 = lambda a, b: b-a
    f3 = lambda a, b: b*a
    f4 = lambda a, b: int(b/a)
    func = {'+': f1, '-': f2, '*': f3, '/': f4} 
    stack = []
    for token in tokens:
        if token in '+-*/':
            # 连续pop出两个值
            a = stack.pop(-1)
            b = stack.pop(-1)   
            stack.append(func[token](a,b))
        else: # 是数字
            stack.append(int(token))
    return stack[-1]
tokens = ["10","6","9","3","+","-11","*","/","*","17","+","5","+"]
print('波兰表示法求后缀表达式的值: ', evalRPN(tokens))

# 小行星碰撞
def asteroidCollision(nums):
    # 相邻数异号则abs小的剔除, 同号则pass [-a,a]则[]
    lens = len(nums)
    stack = []
    i = 0
    while i < lens:   # stack[-1] < 0 or nums[i]已在栈内的向左, 即将加入的向右
        if not stack or stack[-1] < 0 or nums[i] > 0:
            stack.append(nums[i])
        # 不进入上面的if, 则nums[i]<=0 stack[-1]>=0. 即异号会碰撞
        elif stack[-1] <= -nums[i]:  # stack[-1]+nums<=0
            # <0, num负的更多, stack[-1]被撞掉消失, num继续跟新的stack[-1]比较 i不移位
            if stack.pop() < -nums[i]:
                continue
            # else: =0, stack[-1],num均消失, 后面会接上i+=1 
        i += 1
    return stack 
nums = [-2,-1,1,2]
print('小行星碰撞: ', asteroidCollision(nums))

# 划分为k个相等的子集
def canPartitionKSubsets(nums):
    summ = sum(nums)
    if summ % k:
        return False
    per = summ // k
    nums.sort()   
    if nums[-1] > per:
        return False
    n = len(nums)
    @cache
    def dfs(s, p):
        if s == 0:
            return True
        for i in range(n):
            if nums[i] + p > per:
                break
            if s >> i & 1 and dfs(s ^ (1 << i), (p + nums[i]) % per): 
                return True
        return False
    return dfs((1 << n) - 1, 0)

# 所有子集 (可重复)
def subsets(nums):
    res = [[]]
    for i in range(len(nums)):
        for subres in res[:]:
            res.append(subres+[nums[i]])
    return res
# 所有子集 (不可重复)
def subsetsWithDup(nums):
    dic = {}
    for i in nums:
        dic[i] = dic.get(i, 0) + 1
    res = [[]]
    for i, v in dic.items():
        temp = copy.copy(res)  
        for j in res:
            temp.extend(j+[i]*(k+1) for k in range(v))
        res = tempda
    return res

# 零钱兑换需要的硬币个数  最少的硬币数目
def coinChange(coins, amount):
    dp = [amount+100] * (amount+1)  # 当amount小于最小coins则根据这个信息返回0
    dp[0]=0
    for i in range(1, amount+1):
        for coin in coins:
            if i >= coin:
                dp[i] = min(dp[i], dp[i-coin]+1)
    return dp[-1] if dp[-1] != amount+100 else -1
coins, amount = [1, 2, 5], 11
print('零钱兑换: ', coinChange(coins, amount))

# 可以有多少零钱兑换方法 
def change(coins, amount):
    dp = [0 for i in range(amount + 1)]
    dp[0] = 1
    for i in range(len(coins)):   
        for j in range(coins[i], amount+1):  
            dp[j] += dp[j - coins[i]]
    return dp[-1]
amount, coins = 5, [1, 2, 5]
print('零钱兑换几种方法: ', change(coins, amount))

# 完全平方数 拆分为多少个平方之和
def numSquares(n):
    f = [i for i in range(n+1)]
    for i in range(n+1):
        j = 1
        while j*j <= i:
            f[i] = min(f[i], f[i-j*j]+1)
            j+=1
    return f[-1]
print('完全平方数, 可拆为多少个平方数之和: ',  numSquares(13))

# 分割等和子集  分成2个子集
def canPartition(nums):
    summ = 0
    for num in nums:
        summ += num 
    if summ%2 != 0:
        return False 
    sub_sum = summ//2
    dp = [0]*(sub_sum+1)  # 按照背包问题做就可了
    dp[0]=1
    for num in nums:
        for i in range(sub_sum, num-1, -1):
            dp[i] += dp[i-num]
    return dp[-1] != 0
print('分割为两个等和子集: ', canPartition([1,5,2,3]))

# 最后一块石头
'''
x == y 两块石头都会被完全粉碎
如果 x != y, 剩余  y-x
'''
def lastStoneWeightII(stones):
    summ = sum(stones)
    target = summ//2
    dp = [0]*(target+1)
    for stone in stones:
        for i in range(target, stone-1, -1):
            dp[i] = max(dp[i], dp[i-stone]+stone)
    return summ-2*dp[-1]
print('最后一块石头:', lastStoneWeightII([31,26,33,21,40]))

# 盈利计划
def profitableSchemes(n, minProfit, group, profit):
    MOD = 10**9 + 7
    dp = [[0] * (minProfit + 1) for _ in range(n + 1)]
    for i in range(0, n + 1):
        dp[i][0] = 1
    for earn, members in zip(profit, group):
        for j in range(n, members - 1, -1):
            for k in range(minProfit, -1, -1):
                dp[j][k] = (dp[j][k] + dp[j - members][max(0, k - earn)]) % MOD
    return dp[n][minProfit]


# 目标和 数组的各个元素可+-加减, 实现结果等于target 问有几种方法(几种加减的组合)
# 子集a-子集b=target, a+b=summ, -> 2a=summ+target 
def findTargetSumWays(nums, target):
    summ = 0
    lens = len(nums)
    for num in nums:
        summ += num 
    if summ < target or (summ+target)%2 != 0 or summ+target<0: return 0 
    sub_sum = (summ+target)//2  # 继续背包问题解法
    dp = [[0]*(sub_sum+1) for _ in range(lens)]
    dp[0][0] = 1
    if sub_sum >= nums[0]:
        dp[0][nums[0]] += 1
    for i in range(1, lens):
        for j in range(sub_sum+1):
            if j >= nums[i]:
                dp[i][j] = dp[i-1][j] + dp[i-1][j-nums[i]]
            else:
                dp[i][j] = dp[i-1][j]
    return dp[-1][-1]
nums, target = [1,1,1,1,1], 3
print('目标和, 数组各元素可加减实现结果等于target: 求可有几种方法', findTargetSumWays(nums, target))

# 转线最便宜的航班 dp
def findCheapestPrice(n, flights, src, dst, k):
    from collections import defaultdict
    connect = defaultdict(dict)  # (a,b):c 城市a到b花费c
    for a, b, c in flights:
        connect[a][b] = c  
    def dfs(city, remain):  # 更新飞去dst需要的费用
        if city == dst:
            return 0
        if not remain:
            return 2**32-1
        remain -= 1
        ans = 2**32-1
        for nxt in connect[city]:
            ans = min(ans, dfs(nxt, remain) + connect[city][nxt])
        return ans
    
    res = dfs(src, k + 1)
    return res if res != 2**32-1 else -1
n, flights, src, dst, k = 3, [[0,1,100],[1,2,100],[0,2,500]], 0, 2, 0
print('中转最便宜的航班:', findCheapestPrice(n, flights, src, dst, k))

# 0全部在1前面 需要做的最少翻转(0变1,1变0)
def minFlipsMonoIncr(s):
    # dp[i][0]前i个最尾是0的最小翻转次数, dp[i][1]前i个最尾是1的最小翻转次数
    lens = len(s)
    dp = [[0,0] for _ in range(lens)]
    dp[0][0] = 0 if s[0] == '0' else 1
    dp[0][1] = 0 if s[0] == '1' else 1
    for i in range(1, lens):
        dp[i][0] = dp[i-1][0]+(0 if s[i] == '0' else 1)
        dp[i][1] = min(dp[i-1][1], dp[i-1][0])+(0 if s[i] == '1' else 1)
    return min(dp[-1])
s = '010110'
print('0在1前面的最小翻转次数: ', minFlipsMonoIncr(s))

# 子序列的数目  不同的子序列
'''
输入: s = "rabbbit", t = "rabbit"
输出: 3
rabbb_it
rabb_bit
rab_bbit
'''
def numDistinct(s,t):
    ns, nt = len(s), len(t)
    if ns<nt: return 0
    dp = [[0]*(ns+1) for _ in range(nt+1)] # [ij]分别ts
    for i in range(ns+1):
        dp[0][i] = 1
    for i in range(1, nt+1):
        for j in range(1, ns+1):
            if s[j-1]==t[i-1]:
                dp[i][j] = dp[i-1][j-1]+dp[i][j-1]
            else:
                dp[i][j] = dp[i][j-1]
    return dp[-1][-1]
# 优化空间
    n, m = len(s), len(t)
    if n < m: return 0 
    dp = [0]*m
    if s[0] == t[0]:
        dp[0] = 1
    for i in range(1, n):
        for j in range(m-1, 0, -1):
            if s[i] == t[j]:
                dp[j] += dp[j-1]
        if s[i] == t[0]:
            dp[0] += 1
    return dp[-1]
print('子序列的数目: ', numDistinct("babgbag", "bag"))

# dp  字符串交织  s1,s2交织是否能得到s3
def isInterleave(s1,s2,s3):
    n1, n2, n3 = len(s1), len(s2), len(s3)
    if n1+n2 != n3: return False
    dp = [[False]*(n2+1) for _ in range(n1+1)]
    dp[0][0] = True
    for i in range(n1+1):
        for j in range(n2+1):
            if j>0 and dp[i][j-1] and s2[j-1]==s3[i+j-1]:
                dp[i][j] = True 
            if i >0 and dp[i-1][j] and s1[i-1]==s3[i+j-1]:
                dp[i][j] = True   # s12交织的和s3去匹配嘛
    return dp[-1][-1]
s1,s2,s3 = "aabcc", "dbbca", "aadbbbaccc"  # "aabcc", "dbbca", "aadbbcbcac"
print('s12交织匹配是否可得到s3: ',isInterleave(s1,s2,s3))

# 最低票价  
# 三种票价, 一天有效, 7天有效, 一个月有效
# dp[n]=min(dp[n-1]+cost[0], dp[n-7]+cost[1], dp[n-30]+cost[2])
# 当天不需要出行, 就dp[n]=dp[n-1]
def mincostTickets(days, costs):
    trip_days = days[-1]
    dp = [0]*(trip_days+1)
    for trip_day in days:
        dp[trip_day] -= 1  # -1标记好当天回去trip, 后面会判断dp[i]是否等于0的.
    for i in range(1, trip_days+1):
        if dp[i] == 0:  # 不是-1则i天不trip
            dp[i] = dp[i-1]  # 当天不旅行则费用等于dp[i-1]
        else:
            a = dp[i-1]+costs[0]  # 买当天票的费用
            if i-7>=0:  # 之前买了周票?
                b = dp[i-7]+costs[1]
            else:
                b = costs[1]
            if i-30>=0:  # 之前买了月票?
                c = dp[i-30]+costs[2]
            else:
                c = costs[2]
            dp[i] = min(a,b)
            dp[i] = min(c, dp[i])
    return dp[-1] 
days, costs = [1,2,3,4,5,6,7,8,9,10,30,31],  [2,7,15]
print('旅行最小票价: ', mincostTickets(days, costs))


# 以图判树
def validTree(n, edges):
    # 判断是不是树: 是连通图; 不存在环
    from collections import defaultdict
    graph = defaultdict(list)  # 存各个节点的连接关系
    for x,y in edges:
        graph[x].append(y)
        graph[y].append(x)
    vis = set()
    def dfs(i, pre):
        if i in vis: return False
        vis.add(i)
        for j in graph[i]:
            if pre != j and not dfs(j,i):
                return False  # 有环
        return True 
    return dfs(0, None) and len(vis) == n  # len(vis) == n 表示节点都能遍历到是连通图
n, edges = 5, [[0,1],[1,2],[2,3],[1,3],[1,4]] # [[0,1],[0,2],[0,3],[1,4]]
print('判断图是否是树形态: ', validTree(n, edges))

# dp 栈 
# 矩形中最大长方形面积  最大矩形
# 直方图思想做: 第一行看做直方图, 前两行看做直方图, 前三行看做直方图...
def maximalRectangle(matrix):
    def maxheight(height, res):  # 计算直方图中的最大面积通用方法
        stack = [-1]    
        n = len(heights)
        for i, num in enumerate(height):
            while stack[-1] != -1 and height[stack[-1]] > num:  # 栈尾对应的最大高度比num高, 则计算面积
                pre_i = stack.pop()
                res = max(res, (i - stack[-1] - 1) * height[pre_i])
            stack.append(i)
        # 遍历完了height数组, stack中还有元素, 则横向边界可变位n, 边长为: n-stack[-1]-1
        while stack[-1] != -1:
            pre_i = stack.pop()
            res = max(res, (n - stack[-1] - 1) * height[pre_i])
        return res
    if not matrix: return 0
    heights, ans = [0] * len(matrix[0]), 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == '0':
                heights[j] = 0
            elif i > 0 and matrix[i - 1][j] == '0':
                heights[j] = int(matrix[i][j])  # 上一行为0, 则高度重新累计
            else: heights[j] += 1  # 上一行也是1, 则高度直接+1
        ans = maxheight(heights, ans)  # 每一行会更新最大直方图面积
    return ans
matrix = ["10100","10111","11111","10010"]
print('矩阵中的最大面积: ', maximalRectangle(matrix))

# 移除k位, 使剩下数值小 不改变字符顺序
def removeKdigits(nums, k):
    stack = []  # 升序最小堆, 遍历数组遇到更小元素, 把小值替换进来
    for val in nums:
        while stack and k and stack[-1] > val:
            stack.pop()
            k -= 1  # 删除这个较大的栈尾
        stack.append(val)
    if k >0:
        stack = stack[:-k]  # k位没删完, 那就把后面几个大的值直接删掉
    # 去掉字符串左边的0 
    return ''.join(stack).lstrip('0') or '0'
nums, k = "10", 2
print('不换字符顺序, 删除k位使得到最小数: ', removeKdigits(nums, k))

# 最近请求次数  https://leetcode.cn/problems/H8086Q/
class RecentCounter:
    def __init__(self):
        self.arr = []
        self.ptr = 0
    def ping(self, t: int) -> int:
        arr, ptr = self.arr, self.ptr
        n = len(arr) + 1
        arr.append(t)
        while ptr < n and t - 3000 > arr[ptr]:
            ptr += 1
        self.ptr = ptr
        return n - ptr

# 132模式: index顺序: ijk, nums[i] < nums[k] < nums[j]
def find132pattern(nums):
    # 132模式: ijk nums[i] < nums[k] < nums[j]
    lens = len(nums)
    last = float('-inf') # 132中的2 init为最小值
    stack = []  # 升序栈, 存132中的3 
    for i in range(lens-1,-1,-1):
        if nums[i] < last:  # 132中的1出现
            return True 
        # 出现nums[i]大于栈尾值, 则2出现了, 更新2
        while stack and stack[-1] < nums[i]:
            last = stack.pop()    
        stack.append(nums[i])
    return False
nums = [-1,3,2,0]
print('132模式: ', find132pattern(nums))

# 继续栈思路 最长有效括号 hard
def longestValidParentheses(str):
    # 栈做括号匹配, 不可匹配的index置1; 问题就变成, 求最长连续0的长度
    lens = len(s)
    stack, mask = [], [0]*lens
    ll, ans = 0, 0
    for i in range(lens):
        if s[i] == '(':
            stack.append(i)
        else:   # 遇到)了
            if not stack:  # 栈为空表示无(了, 这时又遇到), 无左括号跟ta匹配~ mask=1
                mask[i] = 1 
            else:  # 栈没空, 则花费一个(跟这个)匹配~
                stack.pop() 
    # 遍历完了str, stack中还剩下没配对完的(, 也mask=1
    for ind in stack:
        mask[ind] = 1
    # [0,0,1,1,...0] 剩下的问题就是统计数组中连续0的最长长度
    for i in range(lens):
        if mask[i]:  # =1则得重新0计数
            ll = 0
            continue
        ll += 1
        ans = max(ll, ans)
    return ans 
s = ")()())"
print('最长有效括号: ', longestValidParentheses(s))

# 数据流的第k大
class KthLargest:
    def __init__(k, nums):
        self.k = k
        self.stack = []
        for i in nums: 
            self.stack.append(i)
    def add(val):
        self.stack.append(val)
        self.stack.sort(reverse=True)
        num = 0
        for i in self.stack:
            num = num + 1
            if num == self.k:
                return i

###########################################################
# 抛硬币 
def probabilityOfHeads(prob, target):
    p = [1-prob[0], prob[0]] 
    p += [0] * (target - 1)    # p的长度变成target+1了~ 后面的不知道就都补0
    dp=[p, [0] * (target + 1)] # dp[0]上回投币状态, dp[1]当前投币状态(未知so初始化全为0)
    for p in prob[1:]:
        dp[1][0] = (1-p) * dp[0][0]  # 这次硬币仍朝下
        for i in range(1, target+1):
            dp[1][i] = ((1-p) * dp[0][i] + p * dp[0][i-1])
        dp[0] = dp[1][:]  # 更新上一次的target情况为当前情况
    return dp[0][target]
prob, target = [0.5,0.5,0.5,0.5,0.5], 0
print('抛硬币: ', probabilityOfHeads(prob, target))

# 最小标记代价
# 调整数组的值使各个元素间的差<=target, 调整代价为abs(差)  各元素值都<=100
def MinAdjustmentCost(A, target):
    n = len(A)
    dp = [[1000000000] * 101 for _ in range(n)]
    for i in range(n):
        for j in range(1, 101):
            if i == 0:
                dp[0][j] = abs(j - A[0])
            else:
                left = max(1, j - target)  # 要兼顾左右差值
                right = min(100, j + target)
                for k in range(left, right + 1):
                    dp[i][j] = min(dp[i][j], dp[i - 1][k] + abs(j - A[i]))
    return min(dp[-1])
A, target = [3,5,4,7], 2
print('最小调整代价: ', MinAdjustmentCost(A, target))

# 不相交的线  简单dp就够了不用想太复杂
'''
1   4   2
|    \
|     \   
1   2   4   2条~
''' 
def maxUncrossedLines(nums1, nums2):
    l1, l2 = len(nums1), len(nums2)
    dp = [[0]*(l2+1) for _ in range(l1+1)]
    for i in range(1,l1+1):
        for j in range(1, l2+1):
            if nums1[i-1]==nums2[j-1]:
                dp[i][j] = dp[i-1][j-1]+1
            else:  # 只能回退-1(i-1,j-1)比较, 不然就可能相交~
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]
nums1, nums2 = [2,5,1,2,5], [10,5,2,1,5,2]
print('不相交的线: ', (nums1, nums2))

# dp 两个字符串的最小ASCII删除和
# 两个字符串, 各自可做若干个字符删除(就会积累ascll值), 删除后两个剩余串相等了~
def minimumDeleteSum(s1, s2):
        m,n=len(s1),len(s2)
        dp=[[float('inf')]*(n+1) for _ in range(m+1)]
        dp[0][0]=0
        for i in range(m):
            dp[i+1][0]=dp[i][0]+ord(s1[i])
        for j in range(n):
            dp[0][j+1]=dp[0][j]+ord(s2[j])
        for i in range(1,m+1):
            for j in range(1,n+1):
                if s1[i-1]==s2[j-1]:
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i-1][j]+ord(s1[i-1]),dp[i][j-1]+ord(s2[j-1]))
        return dp[-1][-1]
s1, s2 = "delete", "leet"
print('两个字符串各个做删除获得ascll值, 使剩余串相等: ', minimumDeleteSum(s1, s2))


# dp 粉刷房间  3间房子
def minCost(costs):
    if not costs:
        return 0
    for i in range(1, len(costs)):
        # [i][j] i是房间index, 012是三种颜色
        costs[i][0] += min(costs[i-1][1], costs[i-1][2])
        costs[i][1] += min(costs[i-1][0], costs[i-1][2])
        costs[i][2] += min(costs[i-1][1], costs[i-1][0])
    return min(costs[-1])  # 最后一间房间, 不同012色组合方案的最小值
costs = [[17,2,17],[16,16,5],[14,3,19]]
print('粉刷房间,相邻不可同色, 求costs最小: ', minCost(costs))
# 粉刷房子|| n个房子粉刷, 也是相邻不同色,求代价
def minCostII(costs):
    n = len(costs)  # 房间数
    k = len(costs[0]) # 颜色数
    dp = [0]*k 
    for i in range(n):
        cur_min_cost = [0]*k 
        for j in range(k):
            cur_min_cost[j] = min(dp[:j]+dp[j+1:]) + costs[i][j]
        dp = cur_min_cost
    return min(dp)
costs = [[1,5,3],[2,9,4]]
print('粉刷问题2: ', minCostII(costs))

# dp 正则表达式
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n+1) for _ in range(m+1)]
    dp[0][0] = True
    for j in range(1, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s[i-1] == p[j-1] or p[j-1] == '.':
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':   
                if s[i-1] != p[j-2] and p[j-2] != '.':
                    dp[i][j] = dp[i][j-2]
                else:
                    dp[i][j] = dp[i][j-2] | dp[i-1][j]
    return dp[m][n]
s, p = 'aa', 'a'  # "ab", ".*"
print('dp做法, 正则表达式匹配: ', isMatch(s,p))

# 玩家1取胜否? 只能从两边取  玩家12交替取树 和大的胜 预测赢家
def PredictTheWinner(nums):
    if not nums: return
    lens = len(nums)   # dp[i][j]表示玩家1在ij范围内能取胜的情况
    if lens%2==0:
        return True   # 偶数个数一定玩家1胜 
    dp = [[0 for _ in range(lens)] for _ in range(lens)]
    for i in range(lens):
        dp[i][i] = nums[i]
    for i in range(lens-1):
        dp[i][i+1] = max(nums[i], nums[i+1])
    for x in range(2, lens):  # 相隔至少两个元素,i ~ i+x
        for i in range(lens-x): # 0 ~ lens-x-1
            tmp1 = nums[i] + min(dp[i+2][i+x], dp[i+1][i+x-1]) 
            tmp2 = nums[i+x] + min(dp[i][i+x-2], dp[i+1][i+x-1])
            dp[i][i+x] = max(tmp1, tmp2)
    return dp[0][-1] >= sum(nums)/2
print('玩家1胜利: ', PredictTheWinner([1,5,233,7]))

# 玩游戏 石头游戏 每次可拿前三个中的一个  得分积累  Alice先开始
def stoneGameIII(stoneValue):
    n = len(stoneValue)
    suffix_sum = [0] * (n - 1) + [stoneValue[-1]]
    for i in range(n - 2, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + stoneValue[i]
    f = [0] * n + [0]
    for i in range(n - 1, -1, -1):
        f[i] = suffix_sum[i] - min(f[i+1:i+4])
    total = sum(stoneValue)
    if f[0] * 2 == total:
        return "Tie"  # 平局
    else:
        return "Alice" if f[0] * 2 > total else "Bob"
print('谁赢, ', stoneGameIII([1,2,3,6]))
def winnerSquareGame(n):  # Alice先Bob后
    f = [False] * (n + 1)
    for i in range(1, n + 1):
        k = 1
        while k * k <= i:
            if not f[i - k * k]:
                f[i] = True
                break
            k += 1
    return f[n]
print('拿走任意 非零平方数 个石子: ', winnerSquareGame(17))
# dp  按照字典拆分单词:  s = "leetcode", wordDict = ["leet", "code"] True 
def wordBreak(s, wordDict):
    lens = len(s)
    dp = [False]*(lens+1)
    dp[0] = True
    for i in range(1, lens+1):
        for j in range(i):
            # dp[j]截止到j(不含j)可被拆分的bool情况
            if dp[j] and s[j:i] in wordDict: # s[j: i]为剩下的sub_str
                dp[i] = True 
                break  # 跳出j的循环, 继续循环i 
    return dp[-1]
s, wordDict = "catsandog", ["cats", "dog", "sand", "and", "cat"]
print('拆分单词, 单词拆分: ', wordBreak(s, wordDict))
# 单词替换  单词匹配时, 字符相对顺序不可换  
'''
输入:  dictionary = ["cat","bat","rat"], sentence = "the cattle was rattled by the battery"
输出:  "the cat was rat by the bat"
'''
# 句子中的各个单词, 用dictionary内的sub_word替换
def replaceWords(dictionary, sentence):
    dictionarySet = set(dictionary)
    words = sentence.split(' ')
    for i, word in enumerate(words):
        for j in range(1, len(words)+1):
            if word[:j] in dictionarySet:
                words[i] = word[:j]
                break
    return ' '.join(words)
dictionary, sentence = ["cat","bat","rat"], "the cattle was rattled by the battery"
print('字典单词替换句中词: ', replaceWords(dictionary, sentence))

# 单词拆分2  字符串 拆成words形成句子
def wordBreak(s, wordDict):
    res = []
    # remove_s是在遍历过程中，不断被缩小的s.
    def dfs(words, remove_s):
        if remove_s == '':  # remove_s==''了，证明已经遍历完整个s了~
            res.append(" ".join(words)) # 可以把所有的words转移到res[]中啦~
            return
        for w in wordDict:
            if remove_s[:len(w)] == w:  # 匹配上了单词
                dfs(words + [w], remove_s[len(w):])
    dfs([],s) # 第一个参数[]是已经匹配到了的word，放进list中.
    return res

# 骑士可以8邻域方向走, 每次走两步, 求k次后还留在棋盘内的概率
def knightProbability(n, k, row, column): # row, column是骑士的起点位置
    dp = [[[0] * n for _ in range(n)] for _ in range(k + 1)]
    for step in range(k + 1):
        for i in range(n):
            for j in range(n):
                if step == 0:
                    dp[step][i][j] = 1
                else:
                    for di, dj in ((-2, -1), (-2, 1), (2, -1), (2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2)):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < n and 0 <= nj < n:
                            dp[step][i][j] += dp[step - 1][ni][nj] / 8
    return dp[k][row][column]
print('骑士还在棋盘内: ', knightProbability(3, 2, 0, 0))

# 角矩形的数量
def countCornerRectangles(grid):
    count = collections.Counter()
    ans = 0
    for row in grid:
        for c1, v1 in enumerate(row):
            if v1:
                for c2 in range(c1+1, len(row)):
                    if row[c2]:
                        ans += count[c1, c2]
                        count[c1, c2] += 1
    return ans

# 4键键盘 
def maxA(N):
    dp = [0] * (N + 1)
    for i in range(1, N + 1):
        dp[i] = dp[i-1] + 1 # 先是直接按A
        for j in range(2, i): # j是按ctrl+C的地方
            # i-j是连续按了多少次ctrl+V，+1是因为原来就有dp[j-2]的A在那里，所以总数是dp[j-2] * (i-j+1)
            dp[i] = max(dp[i], dp[j-2] * (i - j + 1))
    return dp[-1]
print('4键键盘', maxA(7))

# 数组错位排序
def findDerangement(n): # D(n)=n * D(n-1) + (-1)^n
    res = 0
    for i in range(n + 1):
        res = (i * res + (-1) ** i) % (10 ** 9 + 7)
    return res

# 分糖果  n个不同的糖放入不同的k个袋子 每个袋子至少一个
def waysToDistribute(n,k):
    dp = [[0 for _ in range(n + 1)] for _ in range(k + 1)]
    for i in range(1, k + 1):
        dp[i][i] = 1        #每个袋子放一个
    for i in range(1, k + 1):       #袋子
        for j in range(i + 1, n + 1):   #糖果数
            #新的糖果，单独一个盒子
            dp[i][j] = dp[i-1][j-1]
            #新的糖果，加入其他的盒子
            dp[i][j] += dp[i][j-1] * i
            dp[i][j] %= 10**9+7
    return dp[k][n]
print('n个不同糖果放入k个不同袋子: ', waysToDistribute(20,5))

# 屏幕可显示句子的数量
'''
一个单词不能拆分成两行
单词在句子中的顺序必须保持不变
在一行中 的两个连续单词必须用一个空格符分隔
'''
def wordsTyping(sentence, rows, cols):
    s = ''
    for word in sentence:
        if len(word) > cols:
            return 0
        s = s + word + ' '
    i = 0
    n = len(s)
    for l in range(rows):
        i += cols
        while s[i % n] != ' ':
            i -=1
        i +=1
    return i // n 
rows, cols, sentence = 3,6,["a", "bcd", "e"]
print('屏幕显示句子: ', wordsTyping(sentence, rows, cols))
# 打家劫舍1  相邻不可偷
def rob(nums):
    if not nums:
        return 0
    n = len(nums)
    if n<=1:
        return nums[0]
    i_2, i_1 = 0, 0  # i-2位置, i-1位置的最大偷窃值
    res = 0
    for i in range(n):
        res = max(i_1, i_2+nums[i])
        i_2 = i_1
        i_1 = res   
    return res
nums = [2,7,9,3,1]
print('相邻位置不可偷的打家劫舍: ', rob(nums))
# 首尾连在一起, 相邻不可偷 
def rob(nums):
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums[0], nums[-1])
    def rob_(a, b, nums):  # 上题的相邻不偷问题
        i_1, i_2, res = 0,0,0
        for i in range(a, b):
            res = max(i_1, i_2+nums[i])
            i_2 = i_1
            i_1 = res 
        return res 
    res1 = rob_(0, n-1, nums) # 偷第一家那就不偷最后一家,
    res2 = rob_(1, n, nums) # 偷最后一家就不偷第一家 
    return max(res2, res1)
nums = [1,2,3,1]
print('首尾连接, 相邻位置不可偷的打家劫舍: ', rob(nums))

# 买股票
# 1. 只交易一次 无手续费
def maxProfit(prices):
    ll = len(prices)
    dp = [0]*ll
    mmin = prices[0]
    for i in range(1, ll):
        mmin = mmin if prices[i] > mmin else prices[i]
        dp[i] = max(dp[i-1], prices[i]-mmin)
    return dp[-1]
prices = [7,6,4,3,1]  # [7,1,5,3,6,4]
print('买股票1: ', maxProfit(prices))
# 无限次购买, 有手续费, 且手上还有股票则不能买入
def maxProfit(prices, free):
    '''
    dp0[i], dp1[i]: (0, i]内最后一次交易是买, 卖
    dp0[i] = max(dp0[i-1], dp1[i-1]-prices[i])  # 没买prices[i], 买了prices[i]
    # 因为再次买入前一定得手上无股票, so得是: dp1[i-1]
    
    dp1[i] = max(dp1[i-1], dp0[i-1]-free+prices[i])  # [i]时候没买卖交易, [i]时候卖了,完成一次买卖需付手续费.
    '''
    dp0, dp1 = -prices[0], 0
    for p in prices[1:]:
        dp0 = max(dp0, dp1-p)
        dp1 = max(dp1, dp0+p-free)
    return dp1
prices, free = [1,3,7,5,10,3], 3 #  [1, 3, 2, 8, 4, 9], 2
print('无限次购买, 有手续费, 且手上还有股票则不能买入, ', maxProfit(prices, free))
# 股票有冷冻期 卖出 冷冻 买入 
def maxProfit(prices):
    n = len(prices)
    if n == 0:
        return 0 
    # dp0, dp1最后一次交易是买入, 卖出
    dp0 = [-prices[0]]*n
    dp1 = [0]*n 
    for i in range(1, n):
        dp0[i] = max(dp0[i-1], -prices[i])
        if i >= 2:
            dp0[i] = max(dp0[i], dp1[i-2]-prices[i])
        dp1[i] = max(dp1[i-1], dp0[i-1]+prices[i])  
    return dp1[n-1]
prices = [1,2,3,0,2]
print('卖出冷冻买入: ', maxProfit(prices))

# 最多两次交易, 没卖出不可买新股
def maxProfit(prices):
    if not prices:
        return 0
    n = len(prices)
    dp = [[[0]*2 for _ in range(3)] for _ in range(n)]
    # dp[i][j][0/1] 0/1代表是否持有股票 ij为第i天交易了第j次
    for j in range(3):
        dp[0][j][0], dp[0][j][1] = 0, -prices[0]
    for i in range(1,n):
        for j in range(3):
            if not j: # j==0 即第i天没有进行交易
                dp[i][j][0] = dp[i-1][j][0]
            else:  # 第i天进行了交易
                dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j-1][1]+prices[i])
                # dp[i-1][j][1]+prices[i] 表示i-1天是有的，i天卖出去了，所以+proces[i]
            dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j][0]-prices[i])
            # dp[i-1][j][0]-prices[i] i-1天没股票，i天买入，所以-prices[i]
    return max(dp[n-1][0][0], dp[n-1][1][0], dp[n-1][2][0])
prices = [7,6,4,3,1]  # [1,2,3,4,5]
print('两次交易卖出前不买入, ', maxProfit(prices))
# 可进行k次交易
def maxProfit(k, prices):
    n = len(prices)
    if not n:
        return 0
    if k >= n//2:
        # 可以一直买入卖出
        res = 0
        for i in range(1, n):
            res += max(0, prices[i]-prices[i-1])
    # dp0为最后一次操作是买入, dp1为最后一次操作是卖出
    dp0 = [-prices[0]]*(k+1)
    dp1 = [0]*(k+1)
    for p in prices[1:]:
        for j in range(1, k+1):
            # dp0[j]: 没买第i支股票
            # or j-1次的最后一下是卖出: dp1[j-1], 第j次是买入 -p
            dp0[j] = max(dp0[j], dp1[j-1]-p)

            # dp1[j]:不卖第i支股票, 则: dp1[i-1][j]
            # or 卖的第i支股, 则dp0[i-1][j] 注意不用j-1, 因为买入+卖出==一次交易. 故: dp0[i-1][j]+p
            dp1[j] = max(dp1[j], dp0[j]+p)
    # 最后肯定要清仓的, dp1.
    return max(dp1[k], 0)
k, prices = 2, [3,2,6,5,0,3]
print('k次股票交易: ', maxProfit(k, prices))

# 数组 排序  数组总和
# 1. 数组部分和==target 不需要连续 可重复数值相加    允许重复选择元素的组合
def func(can, target, path, minV):
    res = []
    for x in range(len(can)):
        diff = target - can[x]
        if diff >= minV:
            res += func(can[x:], diff, path + [can[x]], can[x])
        elif diff == 0:
            res += [path + [can[x]]]
    return res
def combinationSum(candidates, target):
    candidates.sort()   # 先排序 从小往大积累sum
    return func(candidates, target, [], min(candidates))
candidates, target = [2,3,5], 8
print('数组元素相加等于target, 可重复, 无需连续', combinationSum(candidates, target))
# 不可重复取用元素
def combinationSum2(nums, target):
    nums.sort()
    table = [None] + [set() for i in range(target)]
    # [None, set([]), set([]), set([]), set([]), set([]), set([])]   target个set()
    for i in nums:
        if i > target:
            break
        for j in range(target - i, 0, -1):
            table[i + j] |= {elt + (i,) for elt in table[j]}
        table[i].add((i,))
    return list(map(list, table[target]))
nums, target = [12,14,11,15], 26
print('数组元素相加等于target, 不可重复, 无需连续', combinationSum2(nums, target))
# nums中元素存在重复, 每个元素只能用一次, 解集不可出现重复组合
def back_Tracking(nums, target, cur, tmp_res, res):
    if target == 0:
        res.append(tmp_res)
    elif target <0:
        return 
    for i in range(cur, len(nums)):
        if i != cur and nums[i] == nums[i-1]: # 针对nums中有重复元素的
            continue
        back_Tracking(nums, target-nums[i], i+1, tmp_res+[nums[i]], res)
def combinationSum2(nums, target):
    res = []
    nums.sort()  # 排序先
    back_Tracking(nums, target, 0, [], res)
    return res 
nums, target = [2,5,2,1,2], 5# [10,1,2,7,6,1,5], 8
print('回溯 nums中有重复, 子数组和等于target, 不允许重复使用组合内的元素: ', combinationSum2(nums, target))
# 排列组合的数目 转为用dp做  可重复使用元素, res内各个子数组也可重复
'''
输入:  nums = [1,2,3], target = 4
输出:  7
所有可能的组合为:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
'''
def combinationSum4(nums, target):
    dp = [0]*(target+1)
    dp[0] = 1
    for i in range(1, target+1):
        summ = 0
        for num in nums:
            if i-num>=0: summ+=dp[i-num]
        dp[i] = summ
    return dp[target]
nums, target = [1,2,3], 4
print('可重复使用, 可不set的和为target的个数: ', combinationSum4(nums, target))

# 数组相对排序  arr2元素都在arr1中, 对arr1排序保持arr2的顺序, arr2中没有的元素则升序放到末尾
def relativeSortArray(arr1, arr2):
    map_arr1= dict()
    for num in arr1:
        map_arr1[num] = map_arr1.get(num, 0)+1
    remain_nums = [a for a in arr1 if a not in arr2]
    # arr2中没有的排序好, 一会直接放尾部
    remain_nums = sorted(remain_nums)
    res = []
    for num in arr2:
        for i in range(map_arr1[num]):
            res.append(num)
    return res + remain_nums
arr1, arr2= [2,3,1,3,2,4,6,7,9,2,19], [2,1,4,3,9,6]
print('数组相对排序: ', relativeSortArray(arr1, arr2))

# 堆 两个升序数组, 求最小的k对 num1[i]+num2[j]
# 最小堆做
def kSmallestPairs(nums1, nums2):
    m, n = len(nums1), len(nums2)
    ans = []
    pq = [(nums1[i] + nums2[0], i, 0) for i in range(min(k, m))]
    import heapq
    while pq and len(ans) < k:
        _, i, j = heapq.heappop(pq)
        ans.append([nums1[i], nums2[j]])
        if j + 1 < n:
            heapq.heappush(pq, (nums1[i] + nums2[j + 1], i, j + 1))
    return ans
nums1, nums2, k = [1,7,11], [2,4,6], 3
print('两个升序数组组成的k对最小和: ', kSmallestPairs(nums1, nums2))

# 扔鸡蛋  
# 两枚鸡蛋确定f楼(高于f则碎,低于f则不碎) 求需要操作的次数
def twoEggDrop(n):
    cur, total = 1,1
    while total < n:
        total += cur+1
        cur += 1
    return cur 
print('两枚鸡蛋确定f, 求需要操作的次数, ', twoEggDrop(100))
# 同上, 但给k枚鸡蛋, 求需要操作的次数
def superEggDrop(k,n):
    memo = {}
    def dp(k, n):
        if (k, n) not in memo:
            if n == 0:
                ans = 0
            elif k == 1:
                ans = n
            else:
                lo, hi = 1, n
                while lo + 1 < hi:
                    x = (lo + hi) // 2
                    t1 = dp(k - 1, x - 1)
                    t2 = dp(k, n - x)
                    if t1 < t2:
                        lo = x
                    elif t1 > t2:
                        hi = x
                    else:
                        lo = hi = x
                ans = 1 + min(max(dp(k - 1, x - 1), dp(k, n - x))
                                for x in (lo, hi))
            memo[k, n] = ans
        return memo[k, n]
    return dp(k, n)
print('扔k枚鸡蛋: ', superEggDrop(3,14))

# 握手问题
def numberOfWays(numPeople):
    # 每个人与除自己外的人握手 共n//2次握手 握手则连线  求不会线交叉的握手方式
    dp=[0]*(numPeople+1)
    dp[0], dp[2] = 1,1
    for i in range(4, numPeople+1):
        for j in range(2, i+1):
            l = j-2
            r = i-j
            count = (dp[l]*dp[r])%1000000007
            dp[i] = (dp[i]+count)%1000000007
            j += 2
    return dp[-1]
print('握手问题: ', numberOfWays(8))

# 移除k个数使剩余的不同元素最少
def findLeastNumOfUniqueInts(arr, k):
    import collections 
    if not arr:
        return 0
    lens = len(set(arr))
    arr = sorted(collections.Counter(arr).items(), key=lambda x:x[1])
    for key, v in arr:
        if v > k:
            return lens
        k -= v
        lens -= 1
    return lens
print('移除k个数使剩余不同元素最少', findLeastNumOfUniqueInts([4,3,1,1,3,3,2], 3))

# 无重叠区间  移除最少个数区间 使剩余区间无重叠
def eraseOverlapIntervals(intervals):
    if not intervals:
        return 0
    intervals.sort(key=lambda x: x[1])
    n = len(intervals)
    right = intervals[0][1]
    ans = 1
    for i in range(1, n):
        if intervals[i][0] >= right:
            ans += 1
            right = intervals[i][1]
    return n - ans
print('删除最少区间使得无重复区间 贪心: ', eraseOverlapIntervals([[1,2],[2,3],[3,4],[1,3]]))