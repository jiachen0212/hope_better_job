# coding=utf-8

# Day 1
# ChenJ with 33
# 不见 coffee
# 2021.02.16


# No.561  数组拆分  简单
# 排序，然后邻居组合 田忌赛马反思路
def arrayPairSum(nums):
    return sum(sorted(nums)[::2])


# No.563 二叉树坡度  简单
# 深度遍历 dfs, 左右减法+左右根加法
def findTilt(self, root):
    res = 0

    def dfs(root):
        nonlocal res
        if not root:
            return 0
        left = dfs(root.left)
        right = dfs(root.right)
        res += abs(left-right)   # 后续遍历思路，左右子树先做减法
        return root.val + right + left  # 深度遍历实现，每一子树加法计算

    dfs(root)  # run dfs fun
    return res



# No.564 最近回文数  难
def nearestPalindromic(self, n: str) -> str:
    ### 分析1：
    # 数字对称的话，之间的一个(或两个)数字，+1或者-1
    # (非零-1，零+1)
    # 数字不对称的话，需要比较以下几个abs结果：
    # 1. 直接前半部分的逆序
    # 2. 最中间的一个(或两个)数，+1
    # 3. 最中间的一个(或两个)数，-1


    ### 分析2：
    # 就是左半边(n位数长度为奇数，左边多分一位，但这多出的一位不参与后续的回文组合)value 为 a，
    # 分别以 a；a-1；a+1 value 赋值构造回文数

    # 另外特殊的三种情况：(单独处理先：)
     # 小于等于10的数，返回n-1
     # 10的幂，返回n-1
     # 若干个9，返回n+2
     # 11，这个数字比较特殊，返回9
    if int(n)<10:
        return str(int(n)-1)
    if int(n[::-1]) == 1:  # n[::-1] 把字符串n收尾逆序
        # 逆序后再int，若value == 1, 那么之前的n就是10的幂.
        return str(int(n)-1)
    if set(n) == {'9'}:
        # tql, 看是不是只有’9‘这个元素，set()都用上了，厉害!
        return str(int(n)+2)
    if n == "11":
        return '9'

    # 做砍半回文复制：
    l = (len(n)+1)//2   # 需要使用双斜杠，不然报错
    a, b = n[:l], n[l:]  # 这个b设置的很妙，后面会用到len(b)-1
    tmps = [str(int(a)-1), a, str(int(a)+1)]  # a-1, a, a+1
    tmps = [i+i[len(b)-1 ::-1] for i in tmps]  # 这里的len(b)-1用的太巧妙了!!!
    return min(tmps, key=lambda x:abs(int(x)-int(n)) or float('inf'))



# No.565 数组嵌套  中等
# 遍历后的元素，value置零
def arrayNesting(nums):
    res = 0
    for i in range(len(nums)):
        a, count = i, 0
        while (nums[a] >= 0):   # 这个while很精髓!!!
            tmp, nums[a] = nums[a], -1  # 置负数
            a, count = tmp, count+1
        res = max(res, count)
    return res

    # res = 0
    # for i in range(len(nums)):
    #     j, cnt = i, 0
    #     while nums[j] >= 0:
    #         temp, nums[j] = nums[j], -1
    #         print(temp, '===')
    #         j, cnt = temp, cnt+1
    #     res = max(res, cnt)
    # return res
# nums = [0,2,1]
# print(arrayNesting(nums))



# No.566
# 矩阵reshape  简单
def reshpae(nums, r, c):
    tmp = []
    res = []
    for i in range(len(nums)):
        tmp += nums[i]  # 直接加原矩阵的一整行
    print(tmp)
    for i in range(0, len(tmp), c):
        res.append(tmp[i:i+c])
    return res
# print(reshpae([[1,2,3],[4,5,6]], 2, 3))



# No.567 字符串排列  中等
# s1的排列，包含于s2(即是s2的子连续串)
def checkInclusion(self, s1: str, s2: str) -> bool:
    # 不需要列出 s1 的多有排序可能
    # 只需要比较 s1 和 s2 的局部滑窗，这两者的哈希表是否一致即可
    # 哈希表一致，即元素组成成分一致，经过换位排序后，必可一模一样！

    l1, l2 = len(s1), len(s2)
    # s2上用滑窗截取字段，并统计子段的哈希表，和s1的哈希表对比是否一致
    c1 = collections.Counter(s1)  # 统计s1的哈希表
    c2 = collections.Counter()  # 用于统计s2子段的哈希表

    p, q = 0, 0  # s2上的滑窗的起终点
    while q < l2:
        c2[s2[q]] += 1
        if c1 == c2:
            return True
        q += 1
        if q-p+1 > l1:
            c2[s2[p]] -= 1 # 移除左边界的元素
            if c2[s2[p]] == 0: # 删除value为0的key值
                del c2[s2[p]]
            p += 1

    return False




# Day 2
# ChenJ with 33
# 不见 coffee
# 2021.02.17


# No.572 另一个树的子树  简单
# 涉及两层递归：外层递归用queue，层次遍历大的树s；
# 内层递归用dfs，实现比较两棵树是否一致
def isSubtree(self, s, t) -> bool:
    # return str(t) in str(s)  # tai sao 了

    # 层次遍历，用上queue队列，先进后出，遍历s的每一个节点(每个节点check其所有子孙)
    # 怎么check？ 那就用dfs了，检查each节点下的“子孙树”，是否和t一致
    def dfs(node1, node2):
        if not node1 and not node2:
            return True
        if not node1 and node2 or node1 and not node2:
            return False
        if node1.val != node2.val:
            return False
        # 左右子节点各自递归
        b1 = dfs(node1.left, node2.left)
        b2 = dfs(node1.right, node2.right)
        return True if b1 and b2 else False

    # ok以上，dfs结束，实现了两棵树是否一致的check工作
    # 那下面就是外层的递归结构了，用上queue，先进后出，checks的每一个节点
    # 层次遍历：每次出队一个元素，就将该元素的孩子节点加入队列中，直至队列中元素个数为0时，出队的顺序就是该二叉树的层次遍历结果.
    queue = deque([s])
    while queue:
        size = len(queue)
        for i in range(size):
            node_cur = queue.popleft()  # 每此弹出一个节点
            if dfs(node_cur, t):
                return True
            if node_cur.left:
                queue.append(node_cur.left)  # 有左孩子？加入队列！
            if node_cur.right:
                queue.append(node_cur.right)
    return False




# No.575 分糖果  简单
# 可以辅助哈希表思想
# 统计下哈希表先，如果糖果总数的一半value 为a，哈希表key个数为b
# a > b 的话，证明无论怎么分，那个种类最多的子数组，里面都是会有重复的，所以 res直接就是b
# 如果 a < b，证明糖果种类非常的多，还有些类别放不下在子数组里，那么这个res数组的种类就直接是
# 子数组的len，也即a
def distributeCandies(self, candyType) -> int:
    hx_map = collections.Counter(candyType)
    half_nums = len(candyType) // 2
    key_len = len(hx_map)
    if key_len >= half_nums:
        return half_nums
    else:
        return key_len




# No.576 出界的(可行)路径数  中等
# DP 动态规划的题
# 限定了最大的步数值N
# 思路：range(1, N+1)吧，比如出界需要x步，这个x就是1~N范围内的数值
# range遍历每一个x值，累加每一个x值下的路径数目，即可.
# 定义出界条件(ij<0 or ij>m,n)，上下左右移动ij变化方式，以及dp,动态规划实现
def findPaths(self, m: int, n: int, N: int, i: int, j: int) -> int:
    # i,j是题目给定的,初始位置参数，最后return时才会用上..

    tips = [[-1,0],[1,0],[0,-1],[0,1]]  # 上下左右移动
    pre = [[0]*n for i in range(m)]  # 上一步的dp矩阵结果

    for k in range(N):  # k值并不影响什么，只是代表可以走N步
        cur = [[0]*n for i in range(m)]  # 初始化此次N值下的dp矩阵

        for x in range(m):
            for y in range(n):
                # 以上的两层 x y 循坏，是为了得到每一个可能初始位置下的dp矩阵
                # 最后会根据给定的参数 ij，取对应位置上的结果
                # 开始移动
                for tip in tips:
                    x_ = x + tip[0]
                    y_ = y + tip[1]
                    if x_ == -1 or y_ == -1 or x_ == m or y_ == n:  # 出界
                        cur[x][y] += 1
                    else:
                        cur[x][y] += pre[x_][y_]  # “继承”之前的路径数值
        pre = cur  # 更新新的dp矩阵
    return pre[i][j] % 1000000007  # i j这个时候才用上




# N0.581 最短无序连续子数组  中等
# 找出这无序的一段，这一段我们进行排序后，那整个数组s就是升序的了..
# 前后指针移步思路: p,q 为此无序子段的起终点，初始值分别为：0,l-1(没调出来, bug..)
# 正解：左右依次遍历两次，从左往右，记录max值，若nums[i]<max，则此位置p需要调整
# 从右往左，记录min值，若nums[i]>min，则此位置q需要调整
# 子段长度即为：q-p+1  但是这个调试起来好恶心... 我的code提交上去超时了...
def findUnsortedSubarray(nums):
    # 左右依次遍历两次，从左往右，记录max值，若nums[i]<max，则此位置p需要调整
    # 从右往左，记录min值，若nums[i]>min，则此位置q需要调整
    '''
    len_ = len(nums)
    p,q = len_-1, 0
    flag = 0
    for i in range(1,len_):  # 从左向右遍历
        if nums[i] < max(nums[:i]):
            p = i
            flag = 1
    for j in range(len_-1, 0, -1):
        # print(nums[j:], '==', nums[j-1])
        if nums[j-1] > min(nums[j:]):
            q = j-1
            flag = 1
            # print(q, '+++')
    print(p, q)
    if flag:
        return p-q+1
    else:
        return 0
    '''

    # 老老实实把原数组排序吧还是，然后再一位位比较,得到左右边界点.
    nums_ = sorted(nums)
    p, q = 0, len(nums)-1
    while p < q:
        if (nums[p]!=nums_[p]) and (nums[q]!=nums_[q]):
            break
        if nums[p]==nums_[p]:
            p += 1
        if nums[q]==nums_[q]:
            q -= 1
    if q == p:
        return 0
    return q-p + 1
# test:
# nums = [2,6,4,8,10,9,14]
# print(findUnsortedSubarray(nums))




# No.583 编辑距离
'''
if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]+1
else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
'''
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for i in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:  # ij从1开始，i-1和j-1对比，才能更新dp[i][j]的值
                dp[i][j] = dp[i-1][j-1]+1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return m+n-2*dp[-1][-1]  # m+n-2*dp[m][n]
# print(minDistance('sea', 'eat'))




# No.587 安装栅栏  难
# 凸包计算




# No.593 有效的正方形  中等
# 4个点组合出6条线，计算6条线段的可能长度，正方形的话，只会有两个长度值
def validSquare(p1, p2, p3, p4):
    # 首先判断，ABCD 四个点是均不相同的四个点，不然绝对就不是正方向
    if p1!=p2 and p1 !=p3 and p1 !=p4 and p2!=p3 and p2!=p4 and p3!=p4:
        # 4个点，可以组合成6条线段 AB AC AD BC BD CD
        pp = [[p1[0]-p2[0], p1[1]-p2[1]],[p1[0]-p3[0], p1[1]-p3[1]],[p1[0]-p4[0], p1[1]-p4[1]],[p2[0]-p3[0], p2[1]-p3[1]], \
        [p2[0]-p4[0], p2[1]-p4[1]],[p3[0]-p4[0], p3[1]-p4[1]]]  # 6条线段

        length_set = set()
        # 计算6条线段的长度，是正方向的话，只会有2个value值，对角线长和边长，
        # 普通矩形是3个结果，菱形是4个结果，其他四边形就不管了...
        for p in pp:
            length = p[0]*p[0] + p[1]*p[1] # 负负会抵消，故不需要使用abs(p[0/1])*abs(p[0/1])
            length_set.add(length)
        # print(length_set)
        return True if len(length_set) == 2 else False
    else:
        return False
# print(validSquare(p1=[0,0], p2 = [1,1], p3 = [0,0], p4 = [0,0]))




# No.594 最长和谐子序列  简单
# 和谐序列：序列内最大小值间差值为1
# 思路1，想偏了，不需要用上dp动态规划..
# 一维dp问题吧 动态方程: 
'''
if nums[i]-min<1 and max-nums[i]<1:
    dp[i] = dp[i-1]+1
else:
    dp[i] = dp[i-1]
    max = nums[i] if max < nums[i]
    min = nums[i] if min < nums[i]
'''
# 思路2，用上哈希呗
# 思路3，先排序数组，再双指针移动..
# 以下思路2实现：
import collections
def findLHS(nums):
    c = collections.Counter(nums)
    if len(c) == 1:
        return 0  # 数组元素全部相等，这是不存在和谐子序的
    len_ = len(c)
    keys = c.keys()
    maxx = 1
    for i in range(len_):
        for j in range(i+1, len_):
            if abs(keys[i]-keys[j]) == 1:
                res = c[keys[i]]+c[keys[j]]
                maxx = res if res > maxx else maxx
    return maxx





# Day 2021.02.22
# ChenJ sxf shuati

# No.34 中等
# 升序数组中找到target的第一和最后一个位置 时复：log n(得考虑二分法)
def searchRange(self, nums, target):
    if not nums:
        return [-1,-1]
    ll = len(nums)
    def left_index(nums, target):
        l, r = 0, ll-1
        while l<= r:
            mid = (l+r)/2
            if target > nums[mid]:  # 注意边界处理
                l = mid+1
            else:
                r = mid-1
        return l

    def right_index(nums, target):
        l, r = 0, ll-1
        while l<=r:
            mid = (l+r)/2
            if nums[mid] > target:  # 注意边界处理
                r = mid-1
            else:
                l = mid+1
        return r

    le, ri = left_index(nums, target), right_index(nums,target)
    return [le, ri] if (le <= ri) else [-1, -1]




# No.204 小于n的数中，质数的个数  简单
# 思路1：写一个是不是质数的函数【bug,这样写会超时...】
def countPrimes(n):
    # range(n)会超时，考虑把 2 3 5 7 11等这些质数的倍数都直接剔除的做法(由1置0)
    # 质数的倍数，肯定不是质数了
    # 减少需要range的个数..
    if n <= 2:  # 2是第一个质数
        return 0
    res = [1]*n
    res[0], res[1] = 0,0 # 0,1都不是质数
    for i in range(2, int(n**0.5)+1):  # 因子只需要到n的根号值就可以停了..
        # index 2 开始是质数
        # i*i 太巧妙了!!!
        if res[i] == 1:
            res[i*i:n:i] = len(res[i*i:n:i])*[0]
    return sum(res)
# print(countPrimes(10))


# No.801  中等
# A B两个数组，每次换A[i] B[i]
# 使 A B 都变递增 的最小交换次数
# 太秒了, 这个cost的[0,1]设置和更新思想!
def minSwap(A, B):

    # cost[0]代表当前位不换时最小交换次数，
    # cost[1]代表当前位交换时最小交换次数
    cost = [0,1]
    for i in range(1, len(A)):
        if A[i-1]<A[i] and B[i-1]<B[i]: # 直到i为止，AB都还是有序的
            if A[i]>B[i-1] and B[i]>A[i-1]:  # AB的i i-1位还满足这样的，
                # 那其实i位换不换无所谓的，不过可以更新下cost a b 两个位置上的值
                cost = [min(cost), min(cost)+1]  # 当前位置换，那肯定要+1了啊! b=a+1
            else:  # AB的i-1, i位不满足，那这次就是要换i位了, cost b 位+1
                cost[1] += 1
        else:
            # 必须换了啊，因为出现不增序了！！！
            # 更新新的cost:
            # cost[a, b]，a表示不换当前位，那么它的值就是“继承”上一个cost的cost[1] 因为不换自己就是换它的前一位嘛
            # b表示换当前位，那么它的值就是不换上一个的cost[0],加上本次的“换”(数值1)，即cost[0]+1
            # 好绕但是好妙啊！！！
            cost = [cost[1], cost[0]+1]
    return min(cost)




# No.1524 和为奇数的子数组数目  中等
# 注意一个细节，子数组内的index要求连续
# 思路太取巧了！！！
def numOfSubarrays(arr):
    # 直接和是奇数的v1 * 和是偶数的v2
    summ = 0
    ji,ou = 0,1
    for num in arr:
        summ += num
        if summ%2 == 0:
            ou += 1
        else:
            ji += 1
    return (ji*ou)%(10**9+7)  # %10**9+7 是为了避免，ji*ou的值太大，于是做一个%，把商的部分去掉..
    # 求出所有和是奇数的可能性，然后乘以和是偶数的可能性，因为和是偶数直接加进去就好，奇+偶还是奇！
# print(numOfSubarrays([1,2,3,4,5,6,7]))























