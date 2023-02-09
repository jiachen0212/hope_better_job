# coding=utf-8
# baidu 一面
# 0 1 1 2 3  斐波拉切数列

def fun(n):
    if n < 0:
        return
    res = [0,1,1]
    if n <= 2:
        return res[n]
    if n > 2:
        return fun(n-1)+fun(n-2)



def fun2(n):
    a, b = 0,1
    if n==0:
        return a
    if n == 1:
        return b
    while n >= 2:
        a,b = b, a+b
        n -= 1
    return b


for i in range(10):
    print(fun2(i))



'''
import numpy
import cc.ClassA as ccls

import cc.ClassA.fun1

import .cc1.xxx
import cc1.xxx

加点，绝对导入  cc2中import了.cc1，再在另一个地方import cc2, 也可以成功找到import cc1的东西不bug

不加点 就相对导入 cc2导入好了cc1没问题，但别的地方导入cc2 ，走到cc2中的 import cc1.xxx的 就会报错 找不到啊...
'''

# 牛顿迭代法, 开二次方, 精度1e-7  开根号
def mysqrt(x):    
    if x == 0:        
        return 0    
    C, x0 = x, x    
    while True:        
        xi = 0.5 * (x0 + C / x0)        
        if abs(x0 - xi) < 1e-7:            
            break        
        x0 = xi    
    return round(x0, 5)

out = mysqrt(5)
print(out)


# 1. 二分法开根号 开方
def mySqrt_1(x):
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
print(mySqrt_1(5), 'half split')


# 打印等腰三角形
n = 5  # 5行***
for i in range(n):    
    out = (n-i)*' '+(2*i+1)*'*'+(n-i)*' '    
    print(out)


# 积分图计算: 一个二维矩阵, 得到另一个二位矩阵结果: 对应index处是它左上角所有元素+自己本身值 之和
import numpy as np
# 思路, dp: 一边算本行, 一边把除本行之外的积分值累加上
def integral(img):
    h, w = img.shape[:2]  # 3, 6
    integ_graph = np.zeros((h, w), dtype=np.int32)

    for x in range(h):
        # 累加当前行的sum
        cur_col = 0
        for y in range(w):
            cur_col += img[x][y]
            integ_graph[x][y] = integ_graph[x-1][y] + cur_col

    return integ_graph

img = [[2,1,2,3,4,3], 
       [3,2,1,2,2,3],
       [4,2,1,1,1,2]]    
img = np.array(img)    
print(integral(img)) 



# 装饰器例子: 
def my_decorator(func):
    def wrapper():
        print("before fun")
        func()
        print("after fun")
    return wrapper

def say_whee():
    print("Whee!")

# 对say_whee函数做一些额外功能添加, my_decorator()就是这个装饰器函数
say_whee = my_decorator(say_whee)
say_whee()




# bfs广度优先做黑线合并(就是无序数组的做一些条件的筛选合并)
def merge_heixian(xs, ys, hs, x_dis_thres=None, y_dis_thres=None):
    all_merged = []
    lens = len(xs)
    visited = [1]*lens
    
    for cur_index in range(lens):
        if visited[cur_index]:
            stack  = [cur_index]
            merged_hx = [cur_index]
            visited[cur_index] = 0
            while stack:
                # 弹出最后一个x, 则x相关的merged_list要加入stack
                cur_ = stack.pop()
                merged_list = merged_fun(cur_, xs, ys, visited, hs, x_dis_thres, y_dis_thres)
                stack.extend(merged_list)
                merged_hx.extend(merged_list)
            for ind in merged_hx:
                visited[ind] = 0
            all_merged.append(merged_hx)
    return all_merged


# leetcode3 最长不重复子串
# 思路: 遍历一次string, 变量: 最长连续不重复子串的start_index, 这个值在发现前后出现重复元素后, +1更新. 
# 同时, 更新此重复元素的最新出现位置. 以及更新最长子串的长度.
def longest_no_rep_strs(strs):
    res = 0
    if strs == None or len(strs) == 0:
        return res

    s_ind = dict()  # 更新各个元素最末出现的index: 当元素出现重复, 则需更新元素key的value
    res_norep_index = 0 # 从0开始更新是否是最长不重复串的起点
    temp = 0  # 存放临时不重复子串长度值
    for ind, str_ in enumerate(strs):
        # s_ind[str_] >= res_norep_index: 从res_norep_index开始往后扫到了一个重复元素
        if (str_ in s_ind) and s_ind[str_] >= res_norep_index:
            res_norep_index = s_ind[str_] + 1  # 在这个重复元素第一次出现的后一位开始累积
        temp = ind - res_norep_index + 1
        s_ind[str_] = ind 
        res = max(res, temp)

    return res 

print(longest_no_rep_strs('abcabcbb'))


# 有效括号判断
def ok_kuohao(a_list):
    res = []
    for char_ in a_list:
        if char_ == '(':
            res.append(char_)
        elif char_ == ')':
            if res and res[-1] == '(':
                # 弹掉最后一个元素
                res.pop()
            else:
                return False

    if len(res) == 0:
        return True
    else:
        return False


a_lists = ['((()','()))','(())']
for a_list in a_lists:
    print(a_list, ok_kuohao(a_list))



# 接雨水问题
def rain(list_):
    leftmax, rightmax = 0, 0
    left_ind, right_ind = 0, len(list_)-1
    res = 0
    while right_ind > left_ind:
        if list_[left_ind] < list_[right_ind]:
            if list_[left_ind] <= leftmax:
                # 出现洼地, 可积累雨水了
                res += leftmax - list_[left_ind]
            else:
                leftmax = list_[left_ind]
            left_ind += 1
        else:
            if list_[right_ind] <= rightmax:
                # 可积累雨水
                res += rightmax - list_[right_ind]
            else:
                rightmax = list_[right_ind]
            right_ind -= 1

    return res 
print(rain([0,1,0,2,1,0,1,3,2,1,2,1]))


# 一段程序来随机播放 10 首歌曲, 输入10, 输出10
import random
exp = ['a','a','a','b','b', 'a','a','a','b','b']
def gen_no_repet_num(nums):    
    l = len(nums)    
    nums = set(nums)    
    if len(nums) == 1:        
        return False    
    #随机取不重复的数组    
    c = random.sample(nums, len(nums))    
    k = l // len(c)    
    t = l % len(c)
    d = c * k + c[:t]    
    return d
out = gen_no_repet_num(exp)
print(out)



# 公共子串长度, or 求出公共子串  m*n暴力算法
def maxstr(s1,s2):
    m = len(s1)
    n = len(s2)
    dp =[[0 for j in range(n+1)] for i in range(m+1)]

    maxlen = 0
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                maxlen = max(dp[i][j], maxlen)
    
    return maxlen




# 左上角到左下角的最短距离
# dp, 注意最上一行, 最左一列, 的初始化赋值即可
def min_lefttop_rightbottom(M):
    n, m = len(M), len(M[0])
    dp = [[0]*m for i in range(n)]

    dp[0][0] = M[0][0]
    # 最左一列初始化
    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + M[i][0]
    # 最上一行
    for j in range(1, m):
        dp[0][j] = dp[0][j-1] + M[0][j]


    for i in range(1, n):
        for j in range(1, m):  # 左边往右, 还是上边往下.
            dp[i][j] = min(dp[i][j-1], dp[i-1][j]) + M[i][j]

    return dp[-1][-1]

array = [[2,2,1],[2,2,1],[1,1,1]]
print(min_lefttop_rightbottom(array))



def minDistance(self, word1, word2):
    m,n = len(word1),len(word2)
    dp = [[0]*(n+1) for i in range(m+1)]
    # 为什么是m+1和n+1? 因为字符长度从1开始计数

    # 2. 状态方程:
    '''
    if word12 ij位置char相等: dp[i][j]=dp[i-1][j-1]
    if word12 ij位置char不等:
        三种可能，word1删除 插入 替换，取min(dp)
        word1[i]替换成与word2[j]相等，则dp等价于dp[i-1][j-1]
        将word1[i]删除,则dp等价于dp[i-1][j]
        word1插入一个与word[j]相等的char,则dp等价于dp[i][j-1]  # 注意这里别搞错了!
        [i-1][j-1],[i-1][j],[i][j-1]
    '''
    dp[0] = [i for i in range(n+1)]
    for i in range(m+1):
        dp[i][0] = i
    for i in range(1, m+1):
        for j in range(1, n+1):
            # 计数提前了一位, so i-1,j-1代表当前index的字符对比
            if word1[i-1] == word2[j-1]: 
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

    return dp[-1][-1]



# 快排
def quick_sort(nums, left, right):
    if left >= right:
        return nums
    low = left
    high = right 

    # key值贯穿不变
    key = nums[left]

    while left < right:
        while left < right and nums[right] >= key:
            right -= 1
        # 选的key比nums[right]大? 则把这个小的
        nums[left] = nums[right]

        while left < right and nums[left] <= key:
            left += 1
        nums[right] = nums[left]
    nums[left] = key 

    quick_sort(nums, left+1, high)
    quick_sort(nums, low, left-1)

    return nums

print(quick_sort([5,7,8,3,0,-1,10], 0, 6))


a = ''  # None 
# is判断很不严格, 主要a不是明确的None, 均会被放过. 
if a is not None:
    print('1')
# a是个空char, 也会被捕捉到不满足if条件
if a:
    print('2')


# 第10个丑数: 丑数: 只有因子 2 3 5 
def ugly_nums(index):
    uglys = [1]  # 第一个丑数
    # 丑数的三个因子: 2,3,5
    ind2, ind3, ind5 = 0, 0, 0
    num2, num3, num5 = uglys[ind2]*2, uglys[ind3]*3, uglys[ind5]*5
    while len(uglys) < index:
        while num2 <= uglys[-1]:
            ind2 += 1
            num2 = uglys[ind2]*2
        # 跳出本循环, 即找到了一直*2然后大于当期最大丑数的那个数
        while num3 <= uglys[-1]:
            ind3 += 1
            num3 = uglys[ind3]*3
        while num5 <= uglys[-1]:
            ind5 += 1
            num5 = uglys[ind5]*5
        uglys.append(min(num2, num3, num5))
    return uglys[-1]
for index in range(1, 13):
    print('第{}个丑数: {}'.format(index, ugly_nums(index)))



# 1~n内共多少质数, 最后一个丑数是啥.
def countPrimes(n):
    if n<=2:
        return 0

    # res存在是否是质数: 1是,0不是
    res = [1]*n
    res[:2] = [0,0]
    # 遍历到n开平方
    for i in range(2, int(n**0.5)+1):
        # i的唯一因子是自己, 故从i*i: n, 之间的所以数都不是质数了.
        if res[i] == 1:
            # 质数的倍数都质0
            res[i*i: n: i] = len(res[i*i:n:i])*[0]
    # 小于n的所有质数: 
    primes = [a for a in range(n) if res[a] == 1]
    return sum(res), primes[-1], primes

index = 100
print("小于{}的质数有: {}, 最大的那个质数是: {}".format(index, countPrimes(index)[0], countPrimes(index)[1]))



# 顺时针原地旋转矩形/图像
def rotate(matrix):
    n = len(matrix)
    # 1. 上下翻转
    # 2. 对角线翻转
    half_n = n//2
    for i in range(half_n):
        matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
    for i in range(n):
        # 只需要换对角线左侧的元素, range到n的话, 反而会把一些元素又换回去了..
        for j in range(i):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            
    return matrix     


# x的n次幂, x可为小数, 负数, n是整数,可正可负.
class Solution:
    def myPow(self, x, n):
        # 巧妙二分: 幂n除以2变成 res = x^(n/2) * x^(n/2)
        # 注意就是当n为奇数时候, res *= x 
        if n == 0:
            return 1
        if n < 0:
           return 1 / self.myPow(x, -n) 
        if n == 1:
            return x
        half_res = self.myPow(x, n//2)
        # 巧妙二分
        res =  half_res*half_res
        if n%2 == 1:
            res *= x
        return res 
S = Solution()
print('-2.5**3 = ', S.myPow(-2.5, 3))


# 阶层计算后, 末尾多少个0
class Solution(object):
    # 计算因子中有多少个5
    def trailingZeroes(self, n):
        res = 0
        while n>=5:
            res += n//5
            n /= 5  # n超过5*5的话, n/=5之后,n//5还会大于1,故多出的5还是可被计算到res中!
        return int(res)



# 位运算 合集
#1. 数组中均出现了两次, 只有一个数只出现一次, 求这个数.
# a^a = 0, 0^b = b
def singleNumber(nums):
    ll = len(nums)
    res = nums[0] 
    for i in range(1,ll):
        res ^= nums[i]

    return res

# 2. 数组中有俩数只出现一次, 其他均出现两次, 求这俩数.
class Solution:
    def singleNumber(self, nums):
        # 只有一个只出现一次的数字, 则数组从左到右做完一次异或
        # a^a = 0, 0^b = b

        # 同样还是先全部异或一遍数组, a^a=0了, 结果res=b^d
        # bin(res), 可根据ta的某一位1, 将两个数区分开: 因为0^1=1,1^0=1
        # res的某一位1上, a,b肯定是一个1,一个0. 按照这一位0or1, 把数组拆分成两份.
        # 两份中再份内做异或, 就分别求出了两个只出现一次的数. 
        # [出现两次的各个元素肯定被分到同一组的]

        cd = 0
        for num in nums:
            cd ^= num
        need_move_bits = len(bin(cd))-3 # 获得cd结果中最左边那个1需要位移的位数
        # 例: bin(7) = '0b111' od是bin()后的前缀, 再-1就是最左边的那个1. 
        # (选最左的1为了方便,num >> need_move_bits后就只剩下这一位了)
        a, b = 0, 0
        for num in nums:
            # num右移need_move_bits, 查看这一位是0or1
            if (num >> need_move_bits) & 1:
                a ^= num
            else:
                b ^= num

        return a, b 


# 3.数组中均出现了3次, 只有一个数只出现一次, 求这个数.
# x^x = 0, x&(~x) = 0. 这样子3个x就变0了.  
def singleNumber(nums):
    a,b = 0,0
    for x in nums:
        a = ~b & (a^x) 
        b = ~a & (b^x)
    return a  # 注意是返回a!!!


#4. 二进制中1的个数 
def one_nums(n):
    # n&(n-1)可以把n的二进制表示的最后一位1消除(res+1), 直到n中的1全部消除完了, 也就是==0了.
    res = 0
    while n:
        n &= (n-1)  # n == n&(n-1)
        res += 1

    return res 
index = 29
print('{}的二进制中: {}个1'.format(index, one_nums(index)))



# 旋转数组找最小
class Solution:
    def findMin(self, nums):
        # nlog(n) 还得是二分法
        if not nums:
            return 
        l, r = 0, len(nums)-1
        # 说了是无重复的数组, 最后要跳出nums[l],故lr肯定不能相等
        while l < r:
            mid = (l+r)//2
            if nums[mid] > nums[r]:  # 证明前后有更小的, l变大
                l = mid+1
            else: # nums[mid] <= nums[r], 故r从mid开始.
                r = mid 

        return nums[l]


# 每行递增的二位数组, 定位target
class Solution:
    def searchMatrix(self, matrix, target):
        if not matrix or len(matrix[0]) == 0:
            return False
        m, n = len(matrix), len(matrix[0])
        for i in range(m):
            # 在每一行内, 用二分
            l, r = 0, n-1
            while l <= r:
                if matrix[i][r] < target:
                    break # 进行下一行i
                mid = (l+r)//2
                if matrix[i][mid] == target:
                    return True
                elif matrix[i][mid] > target:
                    r = mid-1
                else:
                    l = mid + 1
        return False

# 或者另一种写法:
def searchMatrix(self, matrix, target):
    if not matrix or len(matrix[0]) == 0:
        return False
    n, m = len(matrix), len(matrix[0])
    # 从右上角点开始找, 小了就往下, 大了就往左 
    row, col = 0, m-1
    while row < n and col>=0 and row >= 0 and col < m:
        if matrix[row][col] == target:
            return True
        if matrix[row][col] < target:
            row += 1
        else: # matrix[row][col] > target 
            col -= 1
    return False


# 出现超过数组长度一半次数的数: O(n)遍历一次就够了.
def times_larger_than_half(nums):
    ll = len(nums)
    # 要求O(n)时间复杂度
    res_num = nums[0] # 初始化为第一个元素
    count = 1
    for i in range(1, ll-1):
        if nums[i] == res_num:
            count += 1
        else:
            count -= 1
        if count == 0:
            res_num = nums[i+1]

    return res_num 
print('出现超过数组长度一半的数: ', times_larger_than_half([2,3,4,4,4,4]))


# 最长连续序列, 只要求数值上连续, 不要求在原数组的位置也是连续的. leetcode128
class Solution:
    def longestConsecutive(self, nums):
        # 0(n)遍历一次就得找到.
        # 用了hash字典, 存储每个num对应可以有多长的连续序列
        hash_dict = dict()
        res_max_length = 0
        for num in nums:
            if num not in hash_dict:
                left = hash_dict.get(num-1, 0)
                right = hash_dict.get(num+1, 0)
                cur_length = left+1+right
                res_max_length = max(res_max_length, cur_length)
                # 更新当前num可以有的最大连续序列长度 
                hash_dict[num] = cur_length

                # 巧妙在这俩值: num-left, num+right的更新
                # 因为要value连续, So直接left可以被num减去, right也是同理
                hash_dict[num-left] = cur_length
                hash_dict[num+right] = cur_length
                
        return res_max_length
S = Solution()
print('最长value但index无需连续的序列长度: ', S.longestConsecutive([100,4,200,1,3,2]))



# 数组, 除自身之外的乘积   # 比较ben的写法..
def productExceptSelf(nums):
    lens = len(nums)
    # 左右遍历两次即可
    left = [1]*lens
    right = [1]*lens
    # 左向右遍历, 右向左遍历
    for ind in range(1, lens):
        left[ind] = left[ind-1]*nums[ind-1]
    for ind in range(lens-2, -1, -1):
        right[ind] = right[ind+1]*nums[ind+1]
    res = []
    for i in range(lens):
        res.append(left[i]*right[i])
    return res 


# 更快优雅版: 记录每个元素的左右乘积, 时复空复都O(n)
def productExceptSelf(nums):
    if not nums:
        return nums
    
    lens = len(nums)
    # 遍历一次, 分别积累一个left左向乘积值, 一个right右向乘积值
    # 且res滞后left,right的值一个index位, res左右同时都在做乘积
    res = [1]*lens
    left, right = 1, 1
    for i in range(lens):
        res[i] *= left
        left *= nums[i]
        # 注意此时的right和res[lens-1-i]
        res[lens-1-i] *= right
        right *= nums[lens-1-i]

    return res 
print(productExceptSelf([1,2,3,4]))


# 顺时针旋转打印矩形  螺旋打印矩形
def spiralOrder(matrix):
    if not matrix:
        return 
    res = []
    while matrix:
        # 分四步打印, 第一行, 最右列, 最底行, 最左列
        #1. 第一行
        res += matrix.pop(0)
        #2. 最右列
        if matrix and matrix[0]:    # matrix[0]保证列上还有元素可打印
            for row in matrix:
                res.append(row.pop())  # 每一行的最末元素构成最右列
        #3. 最底行
        if matrix:
            res += matrix.pop()[::-1]  # [::-1]逆序一下右往左打印
        #4. 最左列
        if matrix and matrix[0]:
            for row in matrix[::-1]:  # [::-1]从下行到上行
                res.append(row.pop(0))     # 每行的第一个元素

    return res
print('螺旋顺时针打印矩阵: {}'.format(spiralOrder([[1,2,3],[8,9,4],[7,6,5]])))


# 三角形最小路径和
def minimumTotal(triangle):
    if not triangle:
        return 
    lens = len(triangle)
    # 自底向下, 原地修改三角形的值
    for i in range(lens-1, 0, -1):
        # 4行的三角形, i: 3,2,1, 预留了上一行的减1.
        # 这个j范围, 其实是i的上一行i-1可以有的最大index范围
        for j in range(i):
            # 自底向上, 做加法. 取min([i][j], [i][j+1])相加
            triangle[i-1][j] += min(triangle[i][j], triangle[i][j+1])
    return triangle[0][0]
print('三角形最短路径: ', minimumTotal([[2],[3,4],[6,5,7],[4,1,8,3]]))



# 买股票 系列
#1. 有时间概念, 不限交易次数 leetcode122
def maxProfit(prices):
    # dp[day_index][1/0有无股票]  代表不同day的利润情况
    if not prices:
        return 
    lens = len(prices)
    dp = [[0,0] for i in range(lens)]
    # init dp
    dp[0][0] = 0
    dp[0][1] = -prices[0]  # 第一天买入股票了
    for i in range(1, lens):  # 更新dp[i][0]和dp[i][1]
        # 第i天没有, 可能是i-1天也没有, 或者i-1有但在day_i卖出了
        dp[i][0] = max(dp[i-1][1]+prices[i], dp[i-1][0])
        dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
    # return 在最后一天没有股票的结果
    return dp[-1][0]
print('股票最大利润: {}'.format(maxProfit([7,1,5,3,6,4])))

# 只可两次交易 leetcode123
class Solution:
    def maxProfit(self, prices):
        if not prices:
            return 
        lens = len(prices)
        # dp: [day][交易_index:0,1,2][1/0有无股票]
        dp = [[[0,0] for a in range(3)] for i in range(lens)]

        #init  只是初始化, 给j=0,1,2在第一天的值
        for j in range(3):
            dp[0][j][0] = 0
            dp[0][j][1] = -prices[0]
        
        lens = len(prices)
        # 更新第二天开始的dp[i][j][0]anddp[i][j]][1]
        for i in range(1, lens):
            for j in range(3):      
                if j == 0:
                    dp[i][j][0] = dp[i-1][j][0]
                else: 
                    # 手上无股票: i-1有但是卖了, i-1也没有今天没交易
                    dp[i][j][0] = max(dp[i-1][j][0], dp[i-1][j-1][1]+prices[i])
                
                # 手上有股票: i-1就有,j交易次数不变 or i-1没有今天买入了
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i]) 
        return max(dp[-1][0][0], dp[-1][1][0], dp[-1][2][0])


# 寻找数组峰值, 时复: log(n) 上二分!
class Solution:
    def findPeakElement(self, nums):
        if not nums:
            return 
        ll = len(nums)
        l, r = 0, ll-1
        # 记住是严格大于左右值, 就不允许l<=r
        while l<r:
            mid = (l+r)//2
            if nums[mid] < nums[mid+1]:
                l = mid + 1
            else:
                r = mid   # 注意r值
        return l 


# 数组 组合成最大数字  python2可通
class Solution(object):
    def largestNumber(self, nums):
        if not nums:
            return
        nums_s = map(str, nums)  
        # cmp是排序规则
        sorted_s = sorted(nums_s, cmp=lambda x,y:cmp(int(y+x), int(x+y)))
        return str(int(''.join(a for a in sorted_s)))

S = Solution()
# print(S.largestNumber([3,30,34,5,9]))


# 打家劫舍 系列dp问题

# 相邻的两家不能偷: dp[i] = max(dp[i-1], dp[i-2]+nums[i])
class Solution:
    def rob(self, nums):
        if not nums:
            return
        ll = len(nums)
        if ll < 3:
            return max(nums)
        dp = [0]*ll
        dp[0] = nums[0]
        dp[1] = max(nums[:2])
        for i in range(2, ll):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp[-1]

# 首尾两家算是想临的, 不可同时偷.
class Solution:
    def rob(self, nums):
        if not nums:
            return 
        ll = len(nums)
        if ll < 3:
            return max(nums)
        dp1 = [0]*ll
        dp2 = [0]*ll
        
        # 分抢第一家, 不抢第一家, 两种情况
        # 不抢第一家, 则最后一家可抢, 遍历至ll
        dp1[1] = nums[1]  # dp1[0]=0
        for i in range(2, ll):
            dp1[i] = max(dp1[i-1], dp1[i-2]+nums[i])
        
        # 抢了第一家, 遍历至ll-1
        dp2[0] = nums[0]
        dp2[1] = max(nums[:2])
        for i in range(2, ll-1):
            dp2[i] = max(dp2[i-1], dp2[i-2]+nums[i])

        return max(dp1[-1], dp2[-2])  # 注意是dp[-2], 因为最后一家不能抢值默认是0了..

# 二叉树布局, 直接相连的两家不可同时偷
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rob(self, root):
        if not root:
            return 0
        def helper(root):
            if not root:
                return [0,0]
            left = helper(root.left)  # [带root的val, child_values]
            right = helper(root.right)
            root_lr = root.val + left[1]+right[1]  # left[1]就是只算child不看带root的.
            childs = max(right) + max(left)  # max([带root, lr_不带root])
            return [root_lr, childs]
        res = helper(root)
        return max(res)












