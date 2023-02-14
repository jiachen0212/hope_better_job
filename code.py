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

# 2023.01.31shuati!
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

# 滑动窗: 一边不断r+1扩大窗, 一边遇到重复元素则不断右移左边界缩窗
# 最长不重复连续子串  无重复最长子串
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

# 有障碍物, 左上到右下 可能得路径 
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
                p = p.next # 先把p前移一位,再给这个位置赋予刚刚q的值
                swap(p,q)# 将q的值给p  使得p遍历的节点都小于key
            q = q.next
        swap(head,p)  # 这一步别漏了,把key_ind和之前的head互换 然后分两段使两段均有序
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

# 二叉树最大路径和  dfs 递归
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

# 


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

# 查找集群内的关键连接  leetcode1192
'''
无向图要么环要么链, 有链的话(只有一个连接,), 没入环之前的点都是关键连接  bfs遍历
剩下找两个环之间的唯一通连接, 也是关键连接. 
https://leetcode.cn/problems/critical-connections-in-a-network/solution/python-bu-hui-suan-fa-zhi-hui-dfs-bfs-by-ikqr/
'''  # 算了这个真不会... 有空再回来细想. 

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

# 岛屿周长  0为水 1为陆地 
def islandPerimeter(grid):
    from scipy.signal import convolve2d
    # 左上角有有相近就-2, 上or下相邻则+1
    return int(abs(convolve2d(grid,[[-2,1],[1,0]])).sum())
grid = [[0,1,0,0],[1,1,1,0],[0,1,0,0],[1,1,0,0]]
print('岛屿的周长: ', islandPerimeter(grid))

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

# dp 栈 
# 矩形中最大长方形面积
# 直方图思想做: 第一行看做直方图, 前两行看做直方图, 前三行看做直方图...
def maximalRectangle(matrix):
    def maxheight(height, res):
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
        # 每一行会更新最大直方图面积 
        ans = maxheight(heights, ans)
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

###########################################################
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

# dp  按照字典拆分单词:  s = "leetcode", wordDict = ["leet", "code"] True 
def wordBreak(s, wordDict):
    lens = len(s)
    dp = [False]*(lens+1)
    dp[0] = True
    for i in range(1, lens+1):
        for j in range(i):
            # dp[j]截止到j(不含j)可被拆分的bool情况
            # s[j: i]为剩下的sub_str
            if dp[j] and s[j:i] in wordDict:
                dp[i] = True 
                break  # 跳出j的循环, 继续循环i 
    return dp[-1]
s, wordDict = "catsandog", ["cats", "dog", "sand", "and", "cat"]
print('按照字典拆分string: ', wordBreak(s, wordDict))

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

# 首尾相邻的打家劫舍
