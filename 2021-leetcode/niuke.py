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

# 顺时针旋转图像
def rotate1(A):
    A[:] = zip(*A[::-1])
    return A



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




# No.583 两个字符串的删除操作  中等
# 每次可任意删除word1 or word2中的一个char，使得最后word1==word2,问最少的操作次数
# 即，求解两字符串的最大公共字符长 再len(word1)+len(word2)-len(公共长)
# dp 二维动态规划问题, 重点在状态方程更新：
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
# 一维dp问题吧 动态方程：
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
class Solution(object):
    def countPrimes(self, n):
        # 把质数的倍数直接剔除
        if n<=2:
            return 0
        res = [1]*n
        res[:2] = [0,0]
        # res[i] == 1, 表明是质数
        # 不是质数的话就把res[i]置0!
        # 遍历到n开平方即可
        for i in range(2, int(n**0.5)+1):
            if res[i] == 1:  # 找到了质数i
                # 根据质数i,把它的倍数为都置0.因为这些位置上都不是质数!
                res[i*i:n:i] = len(res[i*i:n:i])*[0]
        return sum(res)


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



# Day 2021.02.23
# sxf shuati

# No.1 扔鸡蛋问题
def dp(K, N): # K个鸡蛋，N层楼
    for i in range(1, N+1):
        # 最坏情况下的最少扔鸡蛋次数
        res = min(res, max(dp(K-1, i-1), # 在第i层碎了，损失一个蛋变k-1，还需尝试i-1层
                           dp(K, N-i))   # 第i层鸡蛋没碎，k个蛋不变，还需尝试N-i层
                           + 1)  # 这个+1是第i层的这次尝试计数
    return res




# No.2 重建二叉树
# 给出前、中序遍历，重构这棵树，返回层次遍历序列.
# -*- coding:utf-8 -*-
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # pre,tin 分别为前序和中序遍历序列
        # pre:根左右、tin:左根右

        if not pre or not tin:
            return None
        root = TreeNode(pre.pop(0))  # 首先得到树的根节点,并且pop掉了, 所以43,44行的pre不用加index截取当前的左右子序列
        # 也可以这么写:
        '''
        root = TreeNode(pre[0]) # 没有把当前的根节点pop掉
        index = tin.index(root.val)
        root.left = self.reConstructBinaryTree(pre[1:1+index], tin[:index])
        root.right = self.reConstructBinaryTree(pre[1+index:], tin[index + 1:])
        '''
        index = tin.index(root.val)  # 找到中序遍历中根节点的位置
        # 开始递归，分别处理左右子树
        root.left = self.reConstructBinaryTree(pre, tin[:index])
        root.right = self.reConstructBinaryTree(pre, tin[index + 1:])
        return root


    # 接着是返回层次遍历结果，作为输出
    # dfs，层次遍历一般用上queue，弹出根，并同时把它下面的左右子都加入queue
    def ccbl(self, root):
        res = []
        if not root:
            return res
        queue = [root]
        while queue:
            node = queue.pop(0)
            res.append(node.val)
            if node.left:
                queue.append(node.left)   # 加入node的左孩子
            if node.right:
                queue.append(node.right)  # 加入node的右孩子
        return res
# test
'''
s = Solution()
pre = [1,2,3,4,5,6,7]
tin = [3,2,4,1,6,5,7]
root = s.reConstructBinaryTree(pre, tin)
print(s.ccbl(root))
'''



# No.3 链表每k个一段，进行反转
# 先遍历一遍链表，求出长度，得到需要反转几段k
# 再进行每段内的反转
class Solution:
    def reverseKGroup(self, head, k):

        myhead = ListNode(0) # 自己虚设一个head前的头结点
        myhead.next = head

        cur = head     # 记头

        ll = 0  # 记录链表总长
        while head:
            ll += 1
            head = head.next

        nums = ll // k  # 需要反转多少段
        if nums == 0:
            return cur

        tail = myhead  # 记尾,先虚设init一下
        for i in range(nums):
            tmp = cur  # 这个tmp就是保存下当前段的头结点，一会会用上
            # 开始每一段内的反转了
            # 涉及 pre cur nxt
            # 因为头结点没有真实的pre，所以单独拿出来做一个22转换
            # 后面k-1次都是33转换
            pre, cur = cur, cur.next
            count = 0
            while count < k-1:
                count += 1
                # 认准后一次反转的cur，一定是上一次的cur.next就好
                # 然后后一次的pre，肯定是当前的cur
                # 剩下的就是 cur.next = pre 咯~

                # pre, cur, cur.next = cur, cur.next, pre  # 这样写就bug
                cur.next, cur, pre  = pre, cur.next, cur   # 这样子ok的！


            # 好了结束了这k个结点的反转, 和下一阶段连接一下
            tail.next = pre  # 这一段的尾巴，接上一段的头(pre)
            tail = tmp  #更新这个记尾结点，为上一段的cur(也即head)

        # 跳出range，把最后不足k长度的"残留"部分接上
        tmp.next = cur  # 也即，上一段的cur的next = cur 类比102行

        return myhead.next

# 简单版: 反转链表
class Solution(object):
    def reverseList(self, head):
        if not head:
            return head
        pre, tmp = None,None
        while head:
            tmp = head.next  # 作为真实移动的指针
            head.next = pre
            pre = head
            head = tmp  # 移动一个位置了，并赋值给head，对应while head
        return pre




# No.4 判断链表是否有环
# 快慢指针: 相遇有环，无法相遇则无环
# No.5 升级版，找环的入口，就是相遇后把fast放到head，然后每次改只有一步(和slow速度一致)
# 下一个相遇点必是环口: f = nc+a s = c+a a是第二次走的，fs步数一样. 这个a必定=kc+b kc抵消在nc中
# 相遇点就是环口了!
class Solution:
    def hasCycle(self, head):
        if not head or not head.next or not head.next.next:
            return False

        # 定义快慢指针
        fast, slow = head.next.next, head.next

        while fast != slow:  # 两指针没相遇接着走，会相遇就代表有环.
            # 这里只针对fast的next进行校验
            # fast走两步，多以得先保证fast.next存在，
            # 但slow.next不能保证fast.next存在，得.next.next
            # fast.next存在可以保证slow.next是一定存在的!
            if fast.next and fast.next.next:
                slow = slow.next
                fast = fast.next.next
            else:
                return False

        return True  # 跳出了while，证明fast slow相遇了

# No.5 找环入口
def detectCycle(self, head):
        if not head or not head.next or not head.next.next:
            return

        # 有无环??
        f,s = head.next.next, head.next
        while f != s:
            if f.next and f.next.next:
                f = f.next.next
                s = s.next
            else:
                return  # 没有环，那也没有环的入口啦 直接return吧
        # 第一次相遇了
        f = head  # f放到开头
        while f != s:
            # 必定会相遇的，所以不需要f.next s.next的none判断
            f = f.next  # f也只走一步了现在
            s = s.next
        # 跳出来了，证明再次相遇，即是环的入口啦~
        return f




# No.6 二叉树是否对称/镜像
# 递归
# 根左右 和 根右左 对比
# line 191: (left.left, right.right) and (left.right, right.left) 很精髓!!!
class Solution:
    def gzy_gyz(self, left, right):
        if not left and not right:  # 到叶结点了无需再对比
            return True
        if not left or not right:   # 左右缺了一边，不对称
            return False
        if left.val != right.val:   # 左右值不等? 再见flase吧..
            return False

        # 开始递归了哈~
        return self.gzy_gyz(left.left, right.right) and self.gzy_gyz(left.right, right.left)

    def isSymmetric(self, root):
        # 前序:根左右 前序对称:根右左
        if not root:
            return True
        return self.gzy_gyz(root.left, root.right)



# No.7  青蛙跳台阶，每次一步或者两步 上n级台阶有几种跳法
# 类似菲布拉契序列  f(n)=f(n-1)+f(n-2)
# n级台阶，先跳一步，剩下的可能性就是f(n-1)的方式跳;
# 先跳两步，剩下那就是f(n-2)的可能方式跳
# 所以是 f(n-1)+f(n-2)

# 递归写法:
'''
# 不过有超时风险哈，牛客上就超时了...
def jumpFloor(number):
    if number <= 2:
        return number
    else:
        return jumpFloor(number-1)+jumpFloor(number-2)
'''

# 非递归写法
def jumpFloor(number):
    # 包括number=0,1 可能性结果是0 1
    # 刚好和number值相等，就直接返回了..
    if number <= 1:
        return number

    # number>=2 index不会越界
    f = [1] * number
    f[0], f[1] = 1, 2
    for i in range(2, number):
        f[i] = f[i-1]+f[i-2]
    return f[-1]




# No.8
# https://www.nowcoder.com/practice/e3769a5f49894d49b871c09cadd13a61?tpId=188&tags=&title=&diffculty=0&judgeStatus=0&rp=1&tab=answerKey
# LRU 缓存结构设计  实现一种数据结构



# No.9 合并两个有序链表，结果仍有序
# 要预设一个头结点前的结点备用[因为tmp.next=l1/l2]
class Solution:
    def mergeTwoLists(self, l1, l2):
        head = ListNode(0)

        tmp = head
        if not l1 and not l2:
            return
        if not l1:
            return l2
        if not l2:
            return l1

        while l1 and l2:
            if l1.val < l2.val:
                tmp.next = l1
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next

        # l1或者l2 有某一个剩下了
        if l1:
            tmp.next = l1
        else:
            tmp.next = l2
        return head.next # 为什么是.next呢? 因为这个head是虚设的结点
        # 要返回它的next，才是合并后的链表的头结点.

# 升级版
# No.10 合并k个有序链表
# 维护最小堆 实现
# 时复n*log(k) n:k个链表的结点总个数
def mergeKLists(self, lists):

    head = ListNode(0)  # 预设一个头结点，其.next指向合并后的链表头
    cur = head  # 后续就是这个cur在移动..

    # 初始化堆
    heapq_ = []
    for sig_list in lists:
        if sig_list:
            heapq_.append((sig_list.val, sig_list))  # 把每一个链表(的头)都加进堆里
    heapq.heapify(heapq_)  # 堆维护

    # 开始根据堆弹出，压入链表结点了
    while heapq_:
        val, node = heapq.heappop(heapq_)
        cur.next = node  # cur的下一个(.next)指向这个新弹出的结点
        cur = cur.next   # cur持续后移
        if node.next:  # 这个链表中还有结点可压入？放进堆里准备维护起来
            heapq.heappush(heapq_,(node.next.val, node.next))
    return head.next





# No.11 字符串做加法  牛客ac了
def solve(s, t):
    if not s and not t:
        return s
    # t = t.lstrip('0')  # 去掉数字首部的无用0

    # 把两字符串补成一样长，方便后续对应位做加法
    lmax = max(len(s), len(t))
    if len(s) < lmax:
        s = '0'*(lmax-len(s)) + s
    else:
        t = '0'*(lmax-len(t)) + t

    jinwei = [0]*(lmax+1)  # 多设置一位，因为最右边的那位加法，进位位值肯定是0的..

    summ = ''
    for i in range(lmax-1, -1, -1):
        cur_sum = int(s[i]) + int(t[i]) + jinwei[i+1]
        if cur_sum > 9:
            jinwei[i] = cur_sum//10
            cur_sum %= 10
        summ += str(cur_sum)

    # 最左边的对应位相加有进位值的话(jinwei[0]!=0)，记得加上
    if jinwei[0]:
        summ += str(jinwei[0])
    return summ[::-1]   # 最后要逆序下
# print(solve('5', '99'))



# No.12 根节点到叶结点和为num
# 递归问题 except_sum, root, cur_path, single_path
class Solution:
    def pathSum(self, root, sum):
        if not root:
            return []
        res = []
        single_path = []
        self.Find(root, sum, res, single_path)
        return res

    # 用于递归的find路径函数
    def Find(self, root, cur_target, res, single_path):
        if not root:
            return
        single_path.append(root.val)  # 不管啥，先当前的父节点加进去
        isLeaf = (not root.left and not root.right)
        if isLeaf and root.val == cur_target:
            res.append(single_path[:])
        if root.left:
            self.Find(root.left, cur_target-root.val, res, single_path)
        if root.right:
            self.Find(root.right, cur_target-root.val, res, single_path)
        single_path.pop()  # 剔除父节点，因为在line365已经把父节点加入了




# No.13 两个栈(先进后出)实现一个队列(先入先出)
# stack1 做push功能，正常进数据
# stack2 做pop功能，需要出数据时，要把stack1内的所有数据依次push进
# stack2，再谈出最顶元素，即可实现先进先出
class Solution:
    def __init__(self):
        self.stack1 = []  # push
        self.stack2 = []  # pop

    def push(self, node):
        self.stack1.append(node)

    def pop(self):
        if self.stack2:
            return self.stack2.pop()  # 2中还有数据，可以弹出
            # 并且是对的先进先出的顺序 直接pop
        elif not self.stack1:
        # stack1 stack2 都是空的..没数据了直接return
            return
        else:
            # stack1有数据，stack2空的,那先把stack1中的全部数据依次弹出，并放入stack2
            # 完了之后，供给stack2依次弹出，此时的顺序也是想要的先进先出
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            return self.stack2.pop()




# No.14 字符串排列组合
# 没咋搞懂..
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.res = []
    def Permutation(self, ss):
        if ss == '':
            return []
        ll = len(ss)
        # 为啥要把ss转成list?
        # 因为string是不可变的,list才可方便后续char换位
        s = []
        for i in range(ll):
            s.append(ss[i])
        self.PermutationHelper(s, 0, ll)
        self.res = list(set(self.res))
        self.res.sort()
        return self.res

    def PermutationHelper(self, ls, pre, end):
        if end <= 0:
            return
        if pre == end-1:
            # 指针走到最后一个元素了已经
            # print(self.res)
            self.res.append(''.join(ls))  # ''.join(ls)把list ls转为str
            # 再append进res
        else:
            for i in range(pre, end):
                self.swap(ls, pre, i)
                self.PermutationHelper(ls, pre+1, end) # 递归
                self.swap(ls, pre, i)  # 上一句的 self.PermutationHelper(ls, pre+1, end)
                # 可能导致pre位置没交换char???

    def swap(self, str_, i, j):
        str_[i], str_[j] = str_[j], str_[i]
# test
# s = Solution()
# s.Permutation('abcd')


# No.15 快排求 数组第k大
class Solution:
    def findKth(self, nums, K):
        # 快排子函数,左边均小于等于key_val，右边大于key_val
        def quick_sort(nums, begin, end):
            index = begin
            key_val = nums[end]
            for i in range(begin, end):
                if nums[i] > key_val:
                    nums[i], nums[index] = nums[index], nums[i]
                    index += 1
            nums[index], nums[end] = nums[end], nums[index]
            return index

        le, ri = 0, len(nums)-1
        index = quick_sort(nums, le, ri)
        # 开始循环
        while index+1 != K:
            if index+1 > K:
                index = quick_sort(nums, le, index-1)
            else:
                index = quick_sort(nums, index +1, ri)
        return nums[index]

# 数组第k大  维护最大堆做
class Solution(object):
    def findKthLargest(self, s, k):
        kk = []
        for i in range(k):
            kk.append(s[i])

        # 另起一个循环,避免重复添加某一元素..
        for i in range(k, len(s)):
            if s[i] > min(kk):  # 把kk中的最小数不断替换掉,加入新的大数
                # kk.pop(kk.index(min(kk)))
                kk.remove(min(kk))
                kk.append(s[i])
        return min(kk)


# No.16 快排 递归
# key_val，左边都小于key_val，右边都大于
# 然后递归，两边都这样处理
# 时复: 最差O(n²)，平均O(nlogn)
def quick_sort(nums, left, right):

    if left>=right:
        return nums

    low = left
    high = right  # left,right作为真正在移动的指针
    # low high 作为保存信息，下面递归时候需要用..

    key = nums[left]
    while left < right:
        while left < right and nums[right]>=key:
            right -= 1
        # nums[right]<key，换到左边去
        nums[left] = nums[right]
        while left < right and nums[left]<=key:
            left += 1
        # nums[left]>key，换到右边去
        nums[right] = nums[left]
    # left>=right了
    # 依旧保持更新nums[left]与key等值
    nums[left] = key

    # ok现在开始递归
    quick_sort(nums, low, left-1)  # 左边也这么快排
    quick_sort(nums, left+1, high) # 右边也这么快排
    return nums
# print(quick_sort([3,5,7,1,0,8,2,4,6],0,8))




# No.17-1 股票卖出最佳时间
# 没有 天数*差价 的概念在里面
# 简单的dp
# dp[i] = max(dp[i-1], nums[i]-min(nums[:i]))
class Solution:
    def maxProfit(self, nums):
        if not nums or len(nums) < 2:
            return 0
        # 没有天数的概念在里面...
        ll = len(nums)
        dp = [0]*ll
        mmin = nums[0]
        for i in range(1, ll):
            mmin = nums[i] if nums[i] < mmin else mmin
            dp[i] = max(nums[i]-mmin, dp[i-1])
        return dp[-1]

# No.17-2
# 可多手交易股票，求最大利润
# 二维dp，dp[i天数][0/1是否持有股票]
class Solution:
    def maxProfit(self, prices):
        ll = len(prices)
        # dp[i][0/1]: 天数index,和 是否持有股票
        dp = [[0]*2 for i in range(ll)]
        dp[0][0] = 0  # 第一天不买入
        dp[0][1] = 0-prices[0] # 第一天买入了

        for i in range(1,ll):
            # 第i天没股票，可能因为i-1天也没有，或者i-1有但是i天卖出了..
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
            # 第i天有股票，可能是第i-1天有，或者第i天买入了
            dp[i][1] = max(dp[i-1][1], dp[i-1][0]-prices[i])
        return dp[-1][0]  # 返回最后一天且手上没有股票的值

# No.17-3
# 只能进行最多两次交易，求最大利润
# 三维dp，第二维记录此次是第几次交易 dp[i天数][j:0、1、2][0/1是否持有股票]
class Solution:
    def maxProfit(self, prices):
        ll = len(prices)
        dp = [[[0]*2 for _ in range(3)] for __ in range(ll)]
        # init
        for j in range(3):
            dp[0][j][0] = 0
            dp[0][j][1] = 0-prices[0]
        for i in range(1,ll):
            for j in range(3):

                # 更新dp[][][0]
                if j == 0:  # j没有交易，那直接就是承接上一天的状态.
                    dp[i][j][0] = dp[i-1][j][0]  # j==0时候不存在j-1,所以单独把j==0拎出来~
                else: # j==1 or 2  # 交易了一次, j-1变j，卖出 or 没交易，上一天也没有股票撒~
                    dp[i][j][0] = max(dp[i-1][j-1][1]+prices[i], dp[i-1][j][0])

                # 更新dp[][][1]
                # 交易次数j不变，前一天就有股票 or 前一天没有股票，交易j-1变j 第i天买入股票
                dp[i][j][1] = max(dp[i-1][j][1], dp[i-1][j-1][0]-prices[i])

        # return交易0、1、2次股票，这三种操作中的最大的受益[dp[-1最后一天][0/1/2][0手上无股]]
        return max(dp[-1][0][0], dp[-1][1][0], dp[-1][2][0])



# No.18 旋转数组 找target、找最小值
# 1.旋转数组找target  搜索旋转排序数组
# 部分有序 先用mid把有序和无序两部分分开
# 要么left:mid有序 要么mid到right有序
# 然后在有序的里面卡target，更改left=mid+1; right=mid-1等..
class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        # 非严格升序数组, 在index-k处旋转, 然后变成两段部分升序的数组
        if not nums:
            return False
        if len(nums) == 1 and nums[0] != target:
            return False
        left, right = 0, len(nums)-1
        while left <= right:
            while left < right and nums[left] == nums[left+1]:
                left += 1
            while left < right and nums[right] == nums[right-1]:
                right -= 1
            
            # 开始二分, mid让两边都有序, 然后看target到底在哪边
            mid = (left+right)//2
            if nums[mid] == target:
                return True
            if nums[mid] >=nums[left]:  # 左边有序,且target小于左侧的最右值, 故mid是该左移的,才可能出发nums[mid]==target
                # 故right值要变小,使mid左移. 
                if nums[left] <= target < nums[mid]:
                    right = mid-1
                else:
                    left = mid +1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False


# 2.旋转数组找最小值
# 尤其注意边界处理: left=mid+1 right=mid
# nums[mid]跟nums[r]比是最靠谱!!!
    def minNumberInRotateArray(self, nums):
        if not nums:
            return 0
        ll = len(nums)
        left,right = 0, ll-1
        while left<right:
            # 为什么不是l<=r，因为可能导致跳不出这个while lr只差1
            mid = (left+right)//2
            if nums[mid] > nums[right]:
                # mid对比r是比较保险的，因为如果nums[mid] > nums[r]，
                # 证明最小值肯定在后半段.
                # 不然，nums[r]一定是最大值的，不存在nums[mid]>nums[r]
                left = mid+1  # mid需要+1再赋值给left，因为nums[mid]处必定不是最小值
            else:
                # nums[mid]<=nums[r]
                # 最小值在前半段
                right = mid  # 因为nums[mid]可能==nums[r]
                # 所以mid直接赋值给right无需+-1
        return nums[left]




# No.19 某数在升序数组中出现的次数
# 统计第一次出现的index1 和 第二次出现的index2

# 方法1: 直接左遍历一次，右遍历一次 [很简单牛客ac了..]
# 方法2: 第一个index(可视为左index)用二分查找; 第二个index(可视为右index)也用二分查找
# 实现下方法2的写法:
class Solution:
    def GetNumberOfK(self, data, k):
        if not data:
            return 0

        def left_index(data, k):
            ll = len(data)
            l,r = 0, ll-1
            while l<=r:
                mid = (l+r)//2
                if data[mid]>=k:
                    r=mid-1
                else:  # data[mid]<k
                    l=mid+1
            return l
        def right_index(data, k):
            ll = len(data)
            l,r = 0, ll-1
            while l<=r:
                mid = (l+r)//2
                if data[mid]<=k:
                    l=mid+1
                else: # data[mid]>k
                    r=mid-1
            return r
        r, l = right_index(data, k), left_index(data, k)
        return r-l+1





# No.20 三个数的最大乘积
# 分析:
# 可以先排序列表，再找最大的三个正数，或者两个最小负数和一个最大正数
# 所以就是，找最大的三个数，和最小的两个数，即可
# a,b, ..., c,d,e  cde和abe比较大小即可!

# 1. 用sort()实现
# class Solution:
#     def solve(self, A):
#         if not A:
#             return
#         ll = len(A)
#         if ll < 2:
#             return
#         A.sort()
#         a,b = A[0]*A[1]*A[-1], A[-3]*A[-2]*A[-1]
#         if a>b:
#             return a
#         else:
#             return b

# 2. 手动记录三个最大值、和、两个最小值
class Solution:
    def solve(self, A):
        if not A or len(A) < 3:
            return
        ll = len(A)
        # 只有三个数
        if ll == 3:
            return A[0]*A[1]*A[2]

        # abc 最大 第二大 第三大
        # de 最小 第二小
        a,b,c,d,e = A[0],A[0],A[0],A[0],A[0]

        for i in range(1,ll):
            # 先处理最小的两个值
            if A[i]<=d:
                e = d
                d = A[i]
            elif A[i]<=e:  # d < A[i] <= e
                e = A[i]

            # 再处理最大的三个值
            if A[i] >= a:  # 要注意cba依次更新值,顺序不能乱了..
                c = b
                b = a
                a = A[i]

            elif A[i] >= b:  # b <= A[i] < a
                c = b
                b = A[i]
            elif A[i] >= c:
                c = A[i]
        return max(a*b*c, d*e*a)



# No.21 最小三角形路径
# 方法1: 直接由下往上修改三角形的value
class Solution:
    def minimumTotal(self, triangle):
        for i in range(len(triangle)-1, 0, -1):
            # i: 3,2,1
            for j in range(i):
                triangle[i-1][j] += min(triangle[i][j], triangle[i][j+1])
        return triangle[0][0]

# 方法2: 用二维dp做，维护一个每行的当前和..
    def minimumTotal(triangle):
        # 在每一行做dp，维护至当前行当前列的和
        ll = len(triangle)
        dp = []  # 初始化一个list先
        dp.append(triangle[0])

        for i in range(1,ll):
            # i==0,1个元素, i==1,2个元素
            cur_line = [0]*(i+1)  # 初始化每一行的和
            cur_line[0] = dp[i-1][0]+triangle[i][0] # 每一行的第一个和，铁定是上一行的第一个和+这一行的第一个值

            # 1~i-1之间的和，则需要dp min比较了..
            for j in range(1, i):
                cur_line[j] = min(dp[i-1][j-1], dp[i-1][j])+triangle[i][j]

            cur_line[i] = dp[i-1][i-1]+triangle[i][i]  # 这一层的最后一个和，铁定是上一层的最后一个和+这层的最后一个值
            dp.append(cur_line) # 加入dp，这一行的cur_line，一会会被当成dp[i-1][]用上~

        # 在dp的最后一行，找出最小值输出即可~
        return min(dp[-1])



# No.22 最长连续序列  难
# nums = [100,4,200,1,3,2]
# 输出：4
# 解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4
# 要求时间复杂度: O(n)  所以得考虑哈希表了哈~
# 维护一个哈希表，key是num值，value是，这个num参与在内的，最长连续串的长度
class Solution(object):
    def longestConsecutive(self, nums):
        if not nums:
            return 0

        # 维护一个哈希表，key是num值，value是，这个num参与在内的，最长连续串的长度
        max_lenght = 1
        mmap = dict()
        for num in nums:
            if num not in mmap: # 哈希表中还没有这个key
                left_number = mmap.get(num-1, 0) # 看看包含num-1这个值的最长连续串的长度, 取不到key就是赋值0
                right_number = mmap.get(num+1, 0)

                # 更新mmap[num]的value
                mmap[num] = left_number+1+right_number
                # 更新维护的最长序列值, 这个值会作为最后的return的
                max_lenght = mmap[num] if mmap[num] > max_lenght else max_lenght

                # 更新这条最长序列的左右边界
                mmap[num-left_number] = mmap[num] # 因为是连续的，且只要nums的元素参与在内，对应的map-value就赋值最长传length
                mmap[num+right_number] = mmap[num] # 右边同理

        # 遍历一遍nums，O(n)时复，即可return
        return max_lenght




# No.23 二叉树，根到叶结点 数值之和
# 递归!!!
# 然后注意一点，递归过程总，原数结点的值会被一直更新: val = val + root*10 (val += root*10)
'''
[1,2,3]
    1
   / \
  2   3
输出: 25   12+13=25
'''
class Solution(object):
    def sumNumbers(self, root):
        if not root:
            return 0
        if not root.left and not root.right:
            return root.val

        # 递归开始，注意会改变原二叉树结点处的值哈~!!!
        all_left, all_right = 0,0
        if root.left:
            root.left.val += root.val*10  # 尤其注意这里的 +=
            # root.left 结点处的值被改变了哈，然后这个结点后续会被递归，一路向左变下去！！
            all_left = self.sumNumbers(root.left)
        if root.right:
            root.right.val += root.val*10  # 同理，改变root.right结点的值哈~
            all_right = self.sumNumbers(root.right)

        return all_right + all_left



# No.24 01矩阵，找出每个元素到最近的0，的距离  return一个同size矩阵结果
# 换个想法：1找最近的0，其实也是: 0在找附近的1，找到后，那个1的位置，赋值原位置+1的值
# 用上bfs，关键在于，0找到1后，把这个1的坐标也加入队列，在此基础上继续寻找附近的1
class Solution(object):
    def updateMatrix(self, matrix):
        if not matrix or not len(matrix[0]):
            return

        m = len(matrix)
        n = len(matrix[0])
        ners = [[-1,0],[1,0],[0,1],[0,-1]]

        queue = []
        for i in range(m):
            for j in range(n):
                if not matrix[i][j]: # matrix[i][j] == 0
                    queue.append([i,j])
                # matrix[i][j] == 1
                else:
                    matrix[i][j] = m+n  # 置一个较大的数，后续ner位置value比较会用到

        # 以上把原矩阵上所有的0的坐标都添加进queue了，接下来就针对0/(已经找到的1)去找附近的1
        while queue:
            cur = queue.pop(0) # 细节，弹出最前面的那个位置
            # 本就是0的那些位置先找，找完后，被0找到的那些1，再去找邻居1.
            for ner in ners:
                x_, y_ = cur[0]+ner[0], cur[1]+ner[1]
                if x_>=0 and x_<m and y_>=0 and y_<n and matrix[x_][y_]>matrix[cur[0]][cur[1]]+1:
                    # matrix[x_][y_]>matrix[cur[0]][cur[1]]+1 很精髓!
                    # 因为上面已经处理过，把matrix中的1都改成了m+n，m+n肯定是>(distance+1)的!
                    # 这个distance是当前位置上，走到最近的1的距离值
                    # +1是因为邻居嘛，距离上就只是+-1这样子
                    # 满足以上条件，证明当前位置找到了一个附近的1了，那么改变这个找到的位置上的值，并且把这个位置加入queue，
                    # 让后续在这个位置基础上再找邻居1
                    matrix[x_][y_] = matrix[cur[0]][cur[1]]+1  # 这个cur可能是原始的0元素位置，也可能是0找到的1，然后在这个1的基础上
                    # 继续找附近的1
                    # 反正就是在原位置的基础上，matrix值+1就对了
                    queue.append([x_,y_])
        return matrix




# No.25 简单 二叉树中两结点最远距离
# 二叉树直径
# 递归，全局变量，找最大的左和右 res=左+右
class Solution(object):
    def diameterOfBinaryTree(self, root):
        self.max_dist = 0  # 全局变量，在递归过程中会一直被更新
        self.Deep(root)
        return self.max_dist
    # 题目一个“坑点”在于，这个路径不一定过根节点的
    # 所以只要一直找左 或者 右；
    # 然后取左右中的最大，并且max=right+left 即可
    def Deep(self, root):
        if not root:
            return -1
        if root.left:
            left = self.Deep(root.left) + 1
        else:
            left = 0
        if root.right:
            right = self.Deep(root.right) + 1
        else:
            right = 0
        self.max_dist = right+left if right+left > self.max_dist else self.max_dist
        return left if left>right else right




# No.26 可被3整除的最大和
# dp
class Solution(object):
    def maxSumDivThree(self, nums):
        dp = [0]*3  # dp[0][1][2]分别为%3余数为012的，数之和
        # 比如有3个数%3==1,那这3个数加起来也可以被3整除;
        # 而不是只有%3==1 + %3==2 的组合才可以
        # 所以一直需要更新余数0 余数1 余数2 这些子序的 和.
        for num in nums:
            # 依次更新 dp[0]dp[1]dp[2]
            if num%3 == 0:
                dp[0] += num
                dp[1] += num
                dp[2] += num
            if num%3 == 1:
                tmp = dp[0]  # 这里为什么要用tmp而不能直接用dp[0]呢?
                # 因为在第一个if后，dp[0]可能值就被改变了，所以得用tmp在后续if中作为原始的dp[0]值参与计算
                # 同理之后的三个if，的顺序也不能随便换，不然会导致前面的某个dp[i]改变了，
                # 后面的if中使用这个dp[i]值就变了，导致程序return 错误!
                if (num+dp[2])%3 == 0:
                    dp[0] = max(dp[0], num+dp[2])
                if (num+dp[1])%3 == 2:
                    dp[2] = max(dp[2], num+dp[1])
                if (num+tmp)%3 == 1:
                    dp[1] = max(dp[1], tmp+num)
            if num%3==2:
                tmp = dp[0]
                if (num+dp[1])%3==0:
                    dp[0] = max(dp[0], dp[1]+num)
                if (num+dp[2])%3 == 1:
                    dp[1] = max(dp[1], num+dp[2])
                if (num+tmp)%3 == 2:
                    dp[2] = max(dp[2], tmp+num)
        return dp[0]


# No.27
# 数组中只出现一次的数字，其他数字都出现过两次
# a^a = 0 0^b = b  数组遍历一遍异或  即可. 
class Solution(object):
    def singleNumber(self, nums):
        ll = len(nums)
        res = nums[0]  # 赋值nums的第一个元素值
        # 之后就和nums的第1~ll-1个元素做异或
        for i in range(1,ll):
            res ^= nums[i]
        return res

# 数组中某元素只出现一次，其他元素都出现三次
# 延用位运算思想，核心在于:
# x^x=0; x&~x=0 那么三个x就变成了0了 就把出现三次的元素都抵消了~
class Solution(object):
    def singleNumber(self, nums):
        a,b = 0,0
        for x in nums:
            a = ~b & (a^x)
            b = ~a & (b^x)
        return a  # 注意是return a.  上行中a先写;b后写.


# 继续，数组中有两个元素只出现一次，其他均出现两次，
# return 出这两个元素
class Solution(object):
    def singleNumber(self, nums):
        # 太妙了，拆分成两段做! 每一段里找出其他都出现两次，只一个出现了一次的那个数
        # 首先还是全员异或一遍，得到 val1 = a^b (假设ab就是出现一次的那两个数)
        # 看这个结果的2进制表示中的非零位(比如我们就看它的最左边那个1，用bin()即可做到!)
        # x^y,只有xy不等时才可能出现值1   0^1=1;1^0=1
        # 所以根据val1中的1 可以把原数组分成两部分，值为0的放一边，值为1的放一边
        # 可以知道，这时候a b 肯定被分得一边一个了
        # 分别对两边做异或，就得到a b啦~
        if not nums:
            return
        val1 = 0  # 0^x = x
        for num in nums:
            val1 ^= num
        a,b=0,0
        yiwei = len(bin(val1))-3 #-3是因为:
        # bin()函数把十进制转为二进制，但前面有ob前缀，例:
        # bin(16): ob10000 最左边的肯定是1，因为bin()会自动把左边的无用0去掉
        # 所以-3也就是减掉了前面的俩ob和最左边的那个1
        # 也就是后续，每一个num要右移yiwei为位，才可以得到每一个num的最左边位上的值，是0还是1
        # 根据01，把数据分到两边去
        # 再就是在两边都做一遍异或, 就可得到a,b
        for num in nums:
            if (num>>yiwei)&1:  # 就是这个位上是1
                a^=num
            else:
                b^=num
        return a, b


# 继续异或!!!
# nums内元素，取值范围是[0,n] nums包含n个元素，求缺失的那个值
# 如: [0,1,3] 缺2
class Solution(object):
    def missingNumber(self, nums):
        # 用异或, 异或初始值设置为: n
        # nums的index: 0~n-1, 假设缺失的是n, 则nums:0~n-1.  
        res = len(nums)
        for index, num in enumerate(nums):
            res ^= index
            res ^= num
        return res


# 继续位运算 与&
# 二进制表示中，位1的个数
# 技巧: n&(n-1)可以起到把二进制表示中最右边的那个1变成0的目的
# 1111&1110 = 1110(1111变1110);  1110&1101 = 1100(1110变1100)
class Solution(object):
    def hammingWeight(self, n):
        res = 0
        while n:
            n &= n-1
            res += 1
        return res


# No.28 找出，出现次数超过数组长度一半的元素
# count计数, 遇到相同的+1.不同的-1
# count等于0时，则换一个元素计数，直到遍历完数组
# 因为出现次数大于序列一半, 结果肯定出现在遍历结束且count>0的值
class Solution(object):
    def majorityElement(self, nums):
        # 摩尔投票法
        count = 1
        res = nums[0]
        for i in range(1,len(nums)-1):
            if nums[i] == res:
                count +=1
            else:
                count -=1
            if count == 0:
                # 为啥是nums[i+1]呢？ 因为肯定不能是nums[i]啊，nums[i]都让count=0了
                res = nums[i+1]
        return res



# No.29 行列递增的二维数组，找出某一target值是否存在
# 二维矩阵，在每一行，这个维度上mid二分
class Solution(object):
    def searchMatrix(self, matrix, target):
        if not matrix or not len(matrix[0]):
            return False
        m,n = len(matrix), len(matrix[0])
        for i in range(m):
            # 对于每一行，在行内进行mid二分
            l,r = 0,n-1
            while l<=r:
                # 注意matrix[i][r] < target的比较要在while内，
                # 因为代码运行过程中r值也在变化着，放在while内方便第一时间跳出
                if matrix[i][r] < target:
                    break # 可以跳出进入下一行了，target比这一行的最大值都大啊!

                # 没有break跳出，那证明target还在这段lr内
                mid = (l+r)//2
                if matrix[i][mid] == target:
                    return True
                elif matrix[i][mid] > target:
                    r = mid-1
                else:
                    l = mid+1
        return False



# No.30 合并两个有序数组，成为一个大的有序数组
# 都放在num1里，预设了num1的空间足够大(>=m+n)
# 【重点是从后往前遍历！！！】【重点是从后往前遍历！！！】
# 因为从前往后遍历的话，把值插入nums1需要调整后续的每一个元素的位置
# 这样显然不行! 并且正好nums1后面是空的0，所以是可以直接置数替换的!
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        i,j,p=m-1,n-1,m+n-1
        while i>=0 and j>=0:
            if nums1[i]>nums2[j]:
                nums1[p] = nums1[i]
                i -= 1
            else:
                nums1[p] = nums2[j]
                j -= 1
            p -= 1
        # 跳出了while，也即，nums1元素走完了，或者nums2中元素走完了
        # 如果nums2走完了，剩下就是nums1,不用管，因为它本身就在nums1里，就结束了呀
        # 如果nums1走完了，那把nums2还没排完的较小值，放到nums1的头部即可
        while j>=0:
            nums1[p] = nums2[j]
            j -= 1
            p -= 1




# No.31 01最大联通域面积
def fun(M, ners):
    m = len(M)
    n = len(M[0])
    mmax = 0
    visited = [[0]*n for i in range(m)]
    for i in range(m):
        for j in range(n):
            # 二维矩阵中的每一个元素，进行bfs搜索.
            count = 0
            if M[i][j] and not visited[i][j]:
                count += 1
                visited[i][j] = 1
                queue = []
                queue.append([i,j])
                while queue:
                    cur = queue.pop(0)  # 弹出最早加入队列的那个元素. pop(0)
                    for ner in ners:
                        x = cur[0]+ner[0]
                        y = cur[1]+ner[1]
                        if (x>=0 and y>=0 and x<m and y<n \
                            and M[x][y] and not visited[x][y]):
                            count += 1
                            visited[x][y] = 1
                            queue.append([x,y])
            # 每一个二维矩阵中的元素bfs后，都比较下mmax要不要更新下~
            mmax = max(mmax, count)
    return mmax



# No.32 乘积最大【连续】子数组
# 思路: 注意要求是连续子数组，所以问题会简单一点点
# 正序遍历一遍，再反序遍历一遍，找出最大值，即可！
class Solution(object):
    def maxProduct(self, nums):
        if not nums:
            return
        ll = len(nums)

        # 正序遍历:
        a = 1
        mmax = nums[0] # 这个mmax是全局的，正反序时候都是在对他修正!
        for i in range(ll):
            a *= nums[i]
            mmax = mmax if mmax>a else a
            if nums[i] == 0:
                # 遇到了0，那只能从新来过，a从新等于1，i+1处再继续乘
                a = 1
        # 反序遍历
        b = 1
        for j in range(ll-1,-1,-1):
            b *= nums[j]
            mmax = mmax if mmax>b else b
            if nums[j] == 0:
                b = 1
        return mmax



# No.33 把数组倒数k个元素搬到数组最前面
# 也称【旋转数组】 原地改变，函数都不用return的!
class Solution(object):
    def rotate(self, nums, k):
        # 因为要求原地翻转，也就是空复完全就是O(1)
        # 思路：k把原数组分成了两段，前ll-k个数a，和后面k个数b
        # 方法就是: a自己翻转下, b自己翻转下,最后整体翻转下，就ok了
        # 实现起来翻译就是：nums的逆=后面k个的逆+前面ll-k个的逆
        # nums[:] = nums[后面的]+nums[前面的]
        nums[:] = nums[-(k%len(nums)):] + nums[:-(k%len(nums))]



# No.34 鸡蛋碎了的问题
class Solution:
    def superEggDrop(self, K: int, N: int) -> int:
        dp = [0] * (K + 1)
        m = 0
        while dp[K] < N:
            m += 1
            for k in range(K, 0, -1):
                dp[k] = dp[k - 1] + dp[k] + 1
                # dp[k-1]鸡蛋碎了, + dp[k]鸡蛋没碎
        return m




# No.35 滑动窗口最大值
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        res, win = [], []
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i-k:
                win.pop(0)
            while win and nums[win[-1]] <= v:
                win.pop()
            win.append(i)
            if i >= k-1:
                res.append(nums[win[0]])
        return res




# No.36  简单
# 把数组中的0移到最后，其他元素相对位置不变
class Solution(object):
    def moveZeroes(self, nums):
        if not nums:
            return
        index = 0
        for num in nums:
            if num != 0:
                nums[index] = num # 遍历到了非零元素，直接替换放在前面
                # index用来从0开始移动位置
                index += 1
        for i in range(index, len(nums)):
            nums[i] = 0  # 有len(nums)-index个0，直接赋值改变这些位置的值为0即可，反正是在nums的尾部了..
        return nums



# No.37 中等
# 返回数组每个除自身外的元素乘积
# 正向、反向各遍历一遍即可~
class Solution(object):
    def productExceptSelf(self, nums):
        if not nums:
            return []
        left,right= 1,1
        ll = len(nums)
        res = [0]*ll
        for i in range(ll):
            res[i] = left
            left *= nums[i]  # left累乘，结果赋值给之后一位的res
            # 也就是，左边乘到自己前面一位为止
            # 同理右边乘到自己后一位为止，下面那个循环会实现
        for j in range(ll-1,-1,-1):
            res[j] *= right   # 注意这里res *= right 而不是直接赋值right，
            # *=是因为它是在乘了left值得基础上，再乘上right右边的值得.!!
            right *= nums[j]
        return res



# No.38 简单
# 求两个数组交集 数组的交集
'''
nums1 = [4,9,5], nums2 = [9,4,9,8,4]
res：[4,9]
'''
class Solution(object):
    def intersect(self, nums1, nums2):
        # 对一个数据用哈希表统计，
        # 然后第二个数组，遍历，查看是否有num1的key，并同时递减计数
        dict1 = dict()
        res = []
        for num in nums1:
            dict1[num] = dict1.get(num, 0)+1 # dict1.get(key,0)
            # dict中存在key，那么就取出它对应的value，取不到，那就新建这个key且value赋值0
        # 开始遍历nums2了
        for num in nums2:
            if num in dict1.keys():
                if dict1[num]:  # dict1中有value个num值，还没被减到0个(还没减完..)
                    res.append(num)
                    dict1[num] -= 1
        return res



# No.39  中等
# 递增的三元子序列
'''
index:(i, j, k),
i < j < k and nums[i] < nums[j] < nums[k]
'''
# 方法1: 暴力ac 时间复杂度好高..
class Solution(object):
    def increasingTriplet(self, nums):
        set(nums)  # 剔除重复元素先
        if not nums or len(nums) < 3:
            return False
        ll = len(nums)
        for i in range(ll-2):
            for j in range(i+1, ll):
                if nums[j]>nums[i]:
                    for k in range(j+1, ll):
                        if nums[k]>nums[j]:
                            return True
        return False

# 方法2:  时复O(n)
class Solution(object):
    def increasingTriplet(self, nums):
        if not nums or len(nums)<3:
            return False
        mmin = 2**31-1
        mid = 2**31-1
        for num in nums:
            if num <= mmin:   # 小于等于的等于,别漏了! <=!!!
            # 记住的技巧: <=的反面才是>， 不然<的反面就是>=了,就不对啊!
                mmin = num
            # num > mmin
            elif num <= mid:
                mid = num   # mmin<num<=mid
            # 出现num > mid，且此时已经存在了mid>mmin
            # 因为mmin,mid这两个值之前一定被更新过(这俩的初始值是2**31-1最大值啊,nums遍历过程
            # 中这两个值肯定会被替换调小的!!)，在前两行的if elif中
            else:
                return True
        return False


# No.40 简单
# 找到相交链表的交点
# p1p2都从各自的头开始走，当某一个走到末尾，就把它放到另一个链表的头，直到相遇即是交点.
# 把l1拆分为a+b,b是与l2共同拥有的那部分; 同理l2拆成c+b
# p1直到与p2相遇，走了l1整个长+l2的前半部分：a+b+c
# p2直到相遇，走了l2整个长+l1的前半部分： c+b+a
# 你看，p1,p2走了相等的长度啊(也就是相遇咯).
# 也就是说等到它俩相遇，就是在交点处了~
class Solution(object):
    def getIntersectionNode(self, headA, headB):
        if not headA or not headB:
            return
        p1,p2 = headA, headB
        while p1!=p2:
            p1 = p1.next if p1 else headB
            p2 = p2.next if p2 else headA
        return p2



# No.41
# 判断是否是回文链表
# 思路1: 转成数值，从前往后算和从后往前算，结果一致则是回文.
# forw, backw, t = 0,0,1
# while head:
#     forw = forw*10+head.val
#     backw += head.val*t
#     t *=10
#     head = head.next
# return forw == backw
# 思路2: 用栈，先进后出，后半段压栈，然后弹出与前半段对比，看是否完全一致.
class Solution(object):
    # 快慢指针找到链表的中点，不需要全部统计整个链表的长度再//2
    def find_mid(self, head):
        fast, slow = head, head
        while fast and fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        return slow
    def isPalindrome(self, head):
        if not head or not head.next:
            return True
        mid_index = self.find_mid(head)

        # 把链表的后半段压入对再弹出，先进后出，然后与前半段比较看是否一致
        queue = []
        mid_index = mid_index.next  # 这一句很精髓
        # 检验过发现，无论原链表是奇或偶，都需要把mid后移一位,再进行压后半段处理!
        while mid_index:
            queue.append(mid_index.val)
            mid_index = mid_index.next
        # 开始遍历前半段，跟栈比较啦~
        p = head
        while queue:
            if queue.pop() != p.val:
                return False
            p = p.next  # 只需要后移p，在上两行queue.pop()中queue自己会一直减小自己的!
        return True



# No.42 行列递增数组找第k小
class Solution(object):
    def kthSmallest(self, matrix, k):
        if not matrix or not matrix[0]:
            return None
        m = len(matrix)
        n = len(matrix[0])
        small,large = matrix[0][0], matrix[-1][-1]
        while small <= large:
            mid = (small+large)//2
            count = 0    # 因为是一个mid对应一个count，所以count定义在while small<=large内
            i,j = m-1,0  # 行:从下往上搜索  列:从左往右
            while i>=0 and j<n:
                if matrix[i][j] <= mid:
                    count += (i+1)  # 因为i的index是0开始
                    j += 1
                else:
                    i -= 1
            # 每次定位一个mid来计数count，
            # mid与k比较，从而调整small、large改变mid
            # 最后return small值.
            if count < k:
                small=mid+1
            else:
                large=mid-1
    return small


###### DP ######
# No.43
# 1. 从矩阵左上角走到右下角的可能路径，(每次只能向下或向右)
class Solution(object):
    def uniquePaths(self, m, n):
        if not m or not n:
            return 0
        dp = [[0]*n for i in range(m)]

        # 这两个初始值别搞错了,
        # 最上一行和最左一列初始值均是1
        dp[0] = [1]*n
        for i in range(m):
            dp[i][0]=1

        # 开始更新dp[][]
        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = dp[i-1][j]+dp[i][j-1] # 两种可能(向下和向右)之和
        return dp[-1][-1]

# 2. 还是矩阵从左上角走到右下角，每个M[i][j]对应一个值
# 只能往下或往右走，问最小路径和
class Solution(object):
    def minPathSum(self, grid):
        # 和三角形最小距离和类似的思路
        if not grid or not len(grid[0]):
            return
        m,n = len(grid), len(grid[0])
        dp = [[0]*n for i in range(m)]
        dp[0][0] = grid[0][0]

        # 注意最上和最左两处的初始化 dp[i-1]+grid[]; dp[j-1]+grid[]
        for i in range(1, m):
            dp[i][0] = grid[i][0] + dp[i-1][0]
        for j in range(1, n):
            dp[0][j] = grid[0][j] + dp[0][j-1]

        for i in range(1,m):
            for j in range(1,n):
                dp[i][j] = min(dp[i-1][j], dp[i][j-1])+grid[i][j]
        return dp[-1][-1]

# 编辑距离
# 3. 实现word1变成word2的最少操作次数.
class Solution(object):
    def minDistance(self, word1, word2):
        # 1. 定义好dp[][]: word1i个字符长,word2j个字符长, word12变一致需要的最少操作次数
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
                if word1[i-1] == word2[j-1]: # 注意这里别搞错了
                    # dp边界是m+1,n+1; 且计数是从1开始. 不做 -1处理的话, word1、2会越界
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1])+1
        return dp[-1][-1]


# No.44 链表排序
# 归并排序链表
class Solution(object):
    def sortList(self, head):
        if not head or not head.next:
            return head
        the_mid = self.get_mid(head)
        l = head
        r = the_mid.next  # 注意是the_mid.next
        the_mid.next = None  # 这一句也注意!
        return self.merge(self.sortList(l), self.sortList(r))

    def get_mid(self, head):
        if not head:
            return head
        fast,slow = head,head
        while fast.next and fast.next.next:  # 注意是while不是if
            fast = fast.next.next
            slow = slow.next
        return slow

    def merge(self, p1, p2):
        tmp = ListNode(0)
        head = tmp
        while p1 and p2:
            if p1.val < p2.val:
                head.next = p1
                p1 = p1.next
            else:
                head.next = p2
                p2 = p2.next
            head = head.next  # head指针向后移动一位,别漏了
        if p1:
            head.next = p1
        else:   # if p2:
            head.next = p2
        return tmp.next


# No.45 寻找数组峰值(大于左右邻居, 时复: log(n)).
class Solution(object):
    def findPeakElement(self, nums):
        ll = len(nums)
        l,r = 0,ll-1
        while l<r:
            mid = (l+r)//2
            if nums[mid]<nums[mid+1]:
                # mid+1起一定存在峰值
                '''
                情况1. nums[mid+2]<nums[mid+1] 那么mid+1就是峰值，两边都比它小
                情况2. numsp[mid+1]>nums[mid+1] 那么mid+2之后可能出现峰值(因为nums[n] = -∞ 最坏最末那么出现峰值)
                '''
                # 所以就调大mid
                l = mid+1
            else:
                r = mid
        return l



# No.46 数组元素组合成最大数
class Solution(object):
    def largestNumber(self, nums):
        s = map(str, nums)  # nums的每个元素变成string 
        l = lambda x,y: int(y+x)-int(x+y) # 定义排序规则l
        res = sorted(s, cmp=l)
        return str(int("".join(i for i in res)))



# No.47 搜索二叉树中的第k小
# 二叉搜索树: 右>根>左
class Solution(object):
    def kthSmallest(self, root, k):
        if not k:
            return
        nodes = []
        self.inOrder(root, nodes)
        if len(nodes)<k:
            return None
        return nodes[k-1]
    # 左父右，中序遍历，且遍历后就是有序的，因为是搜索二叉树，左<父<右
    def inOrder(self, root, nodes):
        if not root:
            return
        if root.left:
            self.inOrder(root.left, nodes)
        nodes.append(root.val)  # 父节点value加入
        if root.right:
            self.inOrder(root.right, nodes)



# No.48 数组 摇摆排序
# 数据：小大小大小大....
# [1,6,1,5,1,4]
# 数组先排序，再分大小两段，再逆序穿插
class Solution(object):
    def wiggleSort(self, nums):
        '''
        先对数组排序，分为大数部分和小数部分，再【降序】穿插排序
        将小数部分的最大放在最左边上，其后紧跟大数部分的最大，避免值相等的问题
        [1,2,4,4,4,6]: [1,2,4],[4,4,6] --> [4,6,2,4,1,4]
        '''
        nums.sort()  # nums升序排序好了
        half = len(nums[::2])  # 半长nums长度
        # nums[::2]:偶数位上的各个数，小数部分的逆序
        # nums[1::2]:奇数位上的数，大数部分的逆序
        nums[::2], nums[1::2] = nums[:half][::-1], nums[half:][::-1]


# No.49 数据流的中位数
# 左边维护一堆小，右边维护一堆大!
def leftmin_rightmax(num, left, right):
    if (len(left)+len(right)) % 2 == 0: # 偶数个
        if left and num < max(left):
            left.append(num)
            num = max(left)
            left.remove(num)
        right.append(num)  # 理论上第一个数是加在right中滴
    else:
        if right and num > min(right):
            right.append(num)
            num = min(right)
            right.remove(num)
        left.append(num)
    return left, right

def main(arrs):
    left, right = [],[]
    for num in arrs:
        left, right = leftmin_rightmax(num, left, right)
    if len(arrs)%2 == 0:
        res = (max(left) + min(right))/2
    else:
        res = min(right)  # 理论上右边会多一个数
    return res


# No.50
# n+1长数组,range[1,n].求出唯一一个重复元素
# 遍历一遍，元素取负，重复的数即可找出
class Solution(object):
    def findDuplicate(self, nums):
        if not nums:
            return
        ll = len(nums)
        tmp = [0]*ll  # 0就是没被遍历，非0就是遍历过了已经，就是重复值
        # num:[1~n] len(nums)=n+1
        for num in nums:
            if tmp[num-1] == 0:# 这个时候把num做index用，所以需要减一下1
                tmp[num-1] = -num  # 置负，作为tag
            else:  # 被遍历过了已经，就是重复值了哈~
                return num


# No.52 最少平方数
# n拆分为若干(最少)个平方数之和
class Solution(object):
    def numSquares(self, n):
        dp = [i for i in range(n+1)]  # 最坏情况，i个1相加等于i，所以每一位init值为i
        for i in range(n+1):
            j = 1
            while j**2 <= n:
                dp[i] = min(dp[i], dp[i-j**2]+1)
                j += 1  # j**2还没大到i，就还可以继续j+1 然后比较 dp[i], dp[i-j**2]+1 取更min值
        return dp[-1]


# No.53 最长递增子序列(index不需要连续, value是连续数组即可)
class Solution(object):
    # 1. 暴力 时复高
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        l = len(nums)
        res = 1
        dp = [1]*l
        for i in range(l):
            for j in range(i):
                if nums[i]>nums[j] and dp[j]+1>dp[i]:
                    dp[i] = dp[j]+1
            res = max(res, dp[i]) # i对应各个index元素位置，每个i维护更新max值res
        return res

    # 二分+dp 来一个num，寻找到合适的插入位置
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        l = len(nums)
        res = 0     # res为最长递增长度
        dp = [0]*l  # dp[i]为: 最长递增长度为i的，最小尾部元素
        # 即dp[i]中的i是(递增串的)长度值，(dp[i]) 是 num-vlaue；有点dict的感觉
        for i in range(l):
            # l、r二分的找到新来的num可以插入到dp中的位置
            # 对每一个num(i)做lr二分查找处理
            l,r = 0,res
            while l<r:
                mid = (l+r)//2
                if dp[mid] < nums[i]:
                    l = mid+1  # num还可以往后放，把l放大
                else:  # dp[mid] >= nums[i]
                    r=mid
            # 跳出while，
            dp[l] = nums[i]  # 把num插入到l位置处
            if l==res:  # l如果!=res,则证明num插入在了0~res之间，那就是说新来的num没有使得
            # max-length变长.
                res += 1
        return res 


# No.54  判断是否是3的幂
class Solution(object):
    def isPowerOfThree(self, n):
        if n == 0:
            return False
        flag = 0
        while n:
            if n%3 == 1:
                flag += 1 # 只可能一次n%3==1,那就是1%3=1 [flag最大值也就只是1了.]
            if n%3 == 2:
                return False
            if flag > 1:
                return False
            n = n // 3
        return True  # 最后1//3==0会跳出while


# No.55 阶乘后,末尾有多少个0
# 算有多少个5即可!!!  10也是5*2得来的!!!
class Solution(object):
    def trailingZeroes(self, n):
        res = 0
        while n>=5:
            res += n//5
            n /= 5
        return int(res)


# No.56
# 颠倒二进制位
# 1010 --> 0101
class Solution:
    def reverseBits(self, n):
        x = 0
        for i in range(32):
            x = x<<1
            x = n&1|x
            n = n>>1
        return x


# No.57 单词拆分1
'''
s = "leetcode"
wordDict = ["leet", "code"]
'''
class Solution(object):
    # dp[i] = 0表示至此位为止不可拆分为若干个单词，dp[i]=1表示可.
    def wordBreak(self, s, wordDict):
        ll = len(s)
        dp = [0]*(ll+1)
        dp[0] = 1

        # 获取wordDict中最长单词的长度
        max_len_word = 0
        for word in wordDict:
            max_len_word = max_len_word if max_len_word > len(word) else len(word)

        for i in range(1, ll+1):  # i可以取值到ll
            # max(0, i-max_len_word) 这样做优化的目的是，当后面i走到很大的时候，
            # j可以不从0开始，而是直接提前到距离i仅max_len_word"差距"的位置
            # 因为就是要判断s[j:i]是否可构成一个单词嘛，最长范围我们已经算出来了:max_len_word啊!
            # dp[j]的j被一下子提前太多担心没有被覆盖遍历? 不怕在i<max_len_word时候，
            # j还是从0开始的... 完美优化!!!
            for j in range(max(0, i-max_len_word), i):  # s[j:i]是否是wordDict中的单词
                if dp[j] and s[j:i] in wordDict:
                    dp[i] = 1
                    break
        return True if dp[-1] == 1 else False

# 单词拆分2 拆分单词并且组合成句子
# dfs+递归  (也有点回溯的意思)
class Solution(object):
    def wordBreak(self, s, wordDict):
        res = []
        # remove_s是在遍历过程中，不断被缩小的s.
        def dfs(words, remove_s):
            if remove_s == '':  # remove_s==''了，证明已经遍历完整个s了~
                # list 2 string 再 2 list
                res.append(" ".join(words)) # 可以把所有的words转移到res[]中啦~
                return
            for w in wordDict:  # 根据不同的w，会有不同的遍历完s的方式，从而产生不同的句子.
                if remove_s[:len(w)] == w:  # 匹配上了单词
                    dfs(words + [w], remove_s[len(w):])
        dfs([],s) # 第一个参数[]是已经匹配到了的word，放进list中.
        return res


# No.58
# 字符串中第一个唯一出现一次的字符
# 【ord() char转ascall, index代替char, 使用list而无需使用dict】【内存耗用更友好~】
class Solution(object):
    def firstUniqChar(self, s):
        if len(s)<1:
            return -1
        # 预设一个256长度的list,使用哈希表把char转为[0,255]范围内的int, (ord()函数实现)
        # 则每个index代表一个char, index上的value则是出现次数
        # 无需dict,节约内存啊~

        hx = [0]*256
        for char in s:
            hx[ord(char)] += 1  # hx list 创建完毕

        for i, char in enumerate(s):
            if hx[ord(char)] == 1:
                return i
        return -1  # s可能没有只出现一次的char, 故-1兜个底



# No.59 分割回文串
'''
s="aab"  --> [[a][a][b]] [[aa][b]]
'''
class Solution(object):
    def partition(self, s):
        # 回溯递归
        ll = len(s)
        if not ll:
            return [[]]
        if ll==1:
            return [[s]]
        tmp = []
        for i in range(1, ll+1):
            left = s[:i]
            right = s[i:]
            if left == left[::-1]:  # 保证left部分已经是回文的了~
                # 回溯法
                right = self.partition(right)
                for j in range(len(right)):
                    tmp.append([left] + right[j])
        return tmp


# No.60 二维矩阵最长递增路径长  diff
class Solution(object):
    def longestIncreasingPath(self, matrix):
        # dfs 就是了..
        if not matrix or not matrix[0]: return 0
        row, col = len(matrix), len(matrix[0])
        lookup = [[0]*col for i in range(row)]
        ners = [[-1, 0], [1, 0], [0, 1], [0, -1]]

        def dfs(i,j):
            if lookup[i][j] != 0:  # 遍历过了此位置且改变了[i][j]位置上的最长递增值
                return lookup[i][j]

            dizeng_length_i = 1
            for x, y in ners:
                tmp_i = x + i
                tmp_j = y + j
                if 0 <= tmp_i < row and 0 <= tmp_j < col and \
                        matrix[tmp_i][tmp_j] > matrix[i][j]:
                    dizeng_length_i = max(dizeng_length_i, 1+dfs(tmp_i, tmp_j))
            lookup[i][j] = max(dizeng_length_i, lookup[i][j])
            return lookup[i][j]  # return更新[i][j]上的最长递增长度值

        return max(dfs(i,j) for i in range(row) for j in range(col))

# No.61  难
# 二叉树最大路径和
class Solution:
    res = float('-inf')
    def maxPathSum(self, root):
        self.getMax(root)
        return self.res

    def getMax(self,root):
        if not root:
            return 0
        # 如果子树路径和为负则应当置0,表示不要子树
        left = max(0, self.getMax(root.left))
        right = max(0, self.getMax(root.right))
        self.res = max(self.res, root.val + left + right)
        return max(left, right) + root.val  # getMax函数是只返回单边的！！！


# No.62 直线上最多的点数  diff
class Solution(object):
    def maxPoints(self, points):
        ll = len(points)
        if ll <3:
            return ll
        res_max = 1
        for i in range(ll):
            same_point = 1
            for j in range(i+1, ll):
                count = 0  # 每一个i的每一个j都从0开始计数
                # 先统计same_point
                if points[i] == points[j]:
                    same_point += 1
                # same_point结束
                else:
                    count += 1  # x首先与任何一个j肯定是在一条直线上的(两点必在一条线啊~)
                    # 然后来开始算 ij直线的diff(斜率)
                    x_diff, y_diff = points[i][0]-points[j][0], points[i][1]-points[j][1]
                    # 在以上diff的基础上,寻找在这条线上的其他点
                    for k in range(j+1, ll):
                        if x_diff*(points[i][1]-points[k][1]) == y_diff*(points[i][0]-points[k][0]):
                            count += 1
                    # 好了基于ij,遍历完了所有的points,更新res_max(i固定,基于每一个j更新)
                    res_max = max(res_max, same_point+count)
            # 对于每一个i,更新res_max
            if res_max > ll/2:  # 大于ll的一半的话，其他点组成的直线不可能数量比res_max多,可直接return了~
                return res_max
        return res_max


# No.63  两数之和
class Solution(object):
    def twoSum(self, nums, target):

        # 可ac但时复高 O(n^2)
        # ll = len(nums)
        # for i in range(ll-1):
        #     for j in range(i+1, ll):
        #         if nums[i] + nums[j] == target:
        #             return [i,j]

        # 用dict, key是num值，value是num对应的index
        # 空复换时复: O(n)和O(n)
        hx = dict()
        for i, num in enumerate(nums):
            if target-num in hx.keys():
                return [i, hx[target-num]]
            else:
                hx[num] = i


# No.64 数组元素替换成右侧最大
class Solution(object):
    def replaceElements(self, arr):
        ll = len(arr)
        cur_max = -1
        # 从后往前遍历,维护一个当前最大
        for i in range(ll-1,-1,-1):
            tmp = arr[i]  # 缓存这个arr[i]值，因为一会arr[i]会被改变..
            arr[i] = cur_max  # 最末时候正好是-1 [原地改变arr的元素值]
            # 更新cur_max
            cur_max = cur_max if cur_max > tmp else tmp
        return arr

# No.65
# 数组中两元素的最大乘积
class Solution(object):
    def maxProduct(self, nums):
        # 求两个最大的数即可(貌似都是正数，不考虑负数的...)
        max1, max2 = 0,0
        ll = len(nums)
        for i in range(ll):
            if nums[i]>max1:
                max2 = max1  # 注意这两行的顺序..
                max1 = nums[i]
            elif nums[i] > max2:
                max2 = nums[i]
        return (max1-1)*(max2-1)


# No.66 打乱数组 shuffle list
# 思路:在前n-1张牌洗好的情况下，第n张牌随机与前n-1张牌的其中一张牌交换/或不换
from random import randint
class Solution:

    def __init__(self, nums):
        self.nums = nums

    def reset(self):
        return self.nums

    def shuffle(self):
        ret = self.nums.copy()
        n = len(ret)
        for i in range(n-1):
            p = randint(i,n-1)
            ret[i],ret[p] = ret[p],ret[i]
        return ret


# No.67  难
# 两个正序数组的中位数
# 难, 没看懂
class Solution:
    def findMedianSortedArrays(self, nums1, nums2):
        n1 = len(nums1)
        n2 = len(nums2)
        if n1 > n2:
            return self.findMedianSortedArrays(nums2,nums1)
        k = (n1 + n2 + 1)//2
        left = 0
        right = n1
        while left < right :
            m1 = left +(right - left)//2
            m2 = k - m1
            if nums1[m1] < nums2[m2-1]:
                left = m1 + 1
            else:
                right = m1
        m1 = left
        m2 = k - m1
        c1 = max(nums1[m1-1] if m1 > 0 else float("-inf"), nums2[m2-1] if m2 > 0 else float("-inf") )
        if (n1 + n2) % 2 == 1:
            return c1
        c2 = min(nums1[m1] if m1 < n1 else float("inf"), nums2[m2] if m2 <n2 else float("inf"))
        return (c1 + c2) / 2


# No.68  数组总和为target,求出所有可能
# 回溯法
class Solution(object):
    def combinationSum2(self, nums, target):
        nums.sort()
        res = []
        def dfs(level, res_current):
            for i in range(level, len(nums)):
                if i > level:
                    if nums[i] == nums[i-1]: continue
                if sum(res_current)+nums[i] == target:
                    res.append(res_current[:] + [nums[i]])
                    return
                elif sum(res_current)+nums[i] < target:
                    dfs(i+1, res_current[:] + [nums[i]])
                else: return
        dfs(0, [])
        return (res)

# No.69 缺失的第一个正数
class Solution:
    def firstMissingPositive(self, nums):
        size = len(nums)
        for i in range(size):
            # 先判断这个数字是不是索引，然后判断这个数字是不是放在了正确的地方
            while 1 <= nums[i] <= size and nums[i] != nums[nums[i]-1]:
                self.__swap(nums, i, nums[i]-1)

        for i in range(size):
            if i+1 != nums[i]:
                return i+1

        return size+1

    def __swap(self, nums, index1, index2):
        nums[index1], nums[index2] = nums[index2], nums[index1]


# No.70 分割等和两个子集
# 背包问题 dp实现
class Solution(object):
    def canPartition(self, nums):
        ll = len(nums)
        if ll < 2:
            return False
        all_sum = sum(nums)
        if all_sum%2 != 0:
            return False  # 无法被2整除,那肯定False无法二等分
        W = all_sum/2  # 开始装包W
        dp = [0]*(W+1)  # dp[i]为装包累加得到的和,最尾为装W的情况
        dp[0] = 1
        for num in nums:
            for i in range(W, num-1, -1):
                # dp[i] = dp[i] + dp[i-num]  # dp[i]表示不拿num, dp[i-num]表示拿num这个数
                dp[i] += dp[i-num]
        return dp[-1] != 0  # dp[-1]不为0即表示可以装包成功


# 类似的,数组三等分: 分成和相等的三份!
class Solution():
    def isEqual_3(self,arr):
        if not arr or len(arr)<3:
            return False
        l = len(arr)
        curSum = [0]*l  # 存放每一个至当前index位置的sum
        curSum[0] = arr[0]
        for i in range(1, l):
            curSum[i] = curSum[i-1]+arr[i]
        tmpSum = curSum[-1] / 3
        if tmpSum*3<curSum[-1]:
            return False    # 证明和是不被3整除的 可以直接false
        # 再开始遍历数组，开始寻找切分点
        for i in range(l):
            if curSum[i] == tmpSum:  # 找到第一个三分点
                if l-1-i<2:  # 后面需要有2段 长度<2肯定不行
                    return False
            if curSum[i] == 2*tmpSum:
                if l-1-i < 1:
                    return False
                else:
                    return True
        return False


# No.71 数组中重复元素
# 部分出现两次,其他仅出现一次
class Solution(object):
    def findDuplicates(self, nums):
        if not nums:
            return
        ll = len(nums)
        res = []
        # 不申请额外的空间存放,那么在置负时候用一个abs()
        for i in range(ll):
            tmp_index = abs(nums[i])-1
            if nums[tmp_index] > 0:
                nums[tmp_index] *= -1
            else:
                res.append(abs(nums[i]))  # 这里abs()是因为这个位置可能在之前被置负了
                # 所以abs()回来!
        return res


# No.72 滑动窗口中位数   diff 没看懂
class Solution:
    def medianSlidingWindow(self, nums, k):
        n = len(nums)
        window = []

        ans = []
        for i in range(n):
            idx = bisect_left(window, nums[i])
            window[idx:idx] = [nums[i]]

            if len(window) > k:
                q = nums[i - k]
                idx = bisect_left(window, q)
                window[idx : idx + 1] = []

            if len(window) == k:
                median = (window[k // 2] + window[(k - 1) // 2]) / 2
                ans.append(median)
        return ans

# No.73 跳跃游戏
# 跳到数组尾部需要的最少步数
class Solution(object):
    def canJump(self, nums):
        l = len(nums)
        if l==1:
            return 0
        q = []
        res = 0
        visited = [False for i in range(l)]
        q.append(0)  # index表示nums中的index
        visited[0] = True
        while q:
            for j in range(len(q)):
                node = q.pop(0)
                for i in range(nums[node], 0, -1): # 从最大开始找有助于加快速度
                # 最大可以跳nums[node]步长.
                    new_index = node+i
                    if new_index >= l-1:
                        return res + 1
                    if not visited[new_index]:
                        visited[new_index] = True
                        q.append(new_index)
            res += 1


# No.74 下一个更大元素
# n的位元素重排,新数>n 即可
class Solution(object):
    def nextGreaterElement(self, n):
        '''
        从右往左遍历先找到第一个比右边小的数字2,然后找到从右往左找到第一个比2大的数字3，
        交换这两个数字，然后421重新排序为124 最后得到2303124
        '''
        nums = [int(x) for x in str(n)]
        if sorted(nums)[::-1] == nums:
            return -1
        m = len(nums)
        for i in range(m-2, -1, -1):
            if nums[i] < nums[i + 1]:
                for j in range(m-1, i, -1):
                    if nums[j] > nums[i]:
                        nums[i], nums[j] = nums[j], nums[i]
                        nums[i+1:] = sorted(nums[i+1:])
                        break
                break
        res = 0
        for i in nums:
            res = 10 * res + i
        return res if res<2**31 else -1


# No.75  和为k的连续子序列 的个数
# 用dict做 (空间换时间,不然暴力做的话,空复就是O(n^2)!)
class Solution(object):
    def subarraySum(self, nums, k):
        count = 0
        if not nums:
            return count
        l = len(nums)
        Sum= 0
        dic = {}
        dic[0]=1
        for i in range(l):
            Sum += nums[i]
            if (Sum-k) in dic:
                count += dic[Sum-k]
            dic[Sum] = dic.get(Sum, 0)+1
        return count


# N0.76
# 有序数组,找k个最接近x的元素
class Solution(object):
    def findClosestElements(self, arr, k, x):
        ll = len(arr)
        l,r = 0,ll-k   # 注意这里r是ll-k
        while l <r:
            mid = (r+l)/2
            if (x-arr[mid] > arr[mid+k]-x):
            # 证明mid选小了 所以l可以后移
                l=mid+1
            else:
                r=mid
        return arr[l:l+k]


# No.77 分割数组位连续子序列
class Solution(object):
    def isPossible(self, nums):
        if not nums or len(nums)<3:
            return False

        l = len(nums)
        dic = {}
        bins = {}   # 存放以nums[i]结尾的，长度>=3的子序列

        for i in range(l):
            bins[nums[i]] = 0

        # 对数组进行遍
        for i in range(l):
            dic[nums[i]] =dic.get(nums[i], 0)+1

        for i in range(l):
            if dic[nums[i]] == 0:
                continue
            dic[nums[i]] -= 1  # 个数自减1
            # 存在以nums[i-1]为末尾元素的子序列
            if nums[i]-1 in bins and bins[nums[i]-1] > 0:
                bins[nums[i]-1] -= 1
                bins[nums[i]] += 1
            # 不存在, 那就从i后面继续找2个数凑下先
            elif nums[i]+1 in dic and nums[i]+2 in dic and dic[nums[i]+1] > 0 and dic[nums[i]+2] > 0:
                dic[nums[i]+1] -= 1
                dic[nums[i]+2] -= 1
                bins[nums[i]+2] += 1   # 以i+2结尾的bins可以+=1
            else:
                return False
        return True


# No.78 柱状图最大矩形面积
# diff 没懂..
class Solution:
    def largestRectangleArea(self, heights):
        ans, s, hs = 0, [0], [0, *heights, 0]
        for i, h in enumerate(hs):
            while hs[s[-1]] > h:
                ans = max(ans, (i - s[-2] - 1) * hs[s.pop()])
            s.append(i)
        return ans

# No.79  二进制求和
class Solution(object):
    def addBinary(self, a, b):
        if a == '0':
            return b
        if 'b' == '0':
            return a
        llen = max(len(a), len(b))
        if len(a) > len(b):
            b = b.zfill(llen)  # zfill()左边补0操作.
        if len(a) < len(b):
            a = a.zfill(llen)

        res = [0] * (llen+1)
        jw = 0
        for i in range(llen-1, -1, -1):
            tmp = int(a[i]) + int(b[i]) + jw
            jw = tmp / 2
            res[i+1] = tmp % 2
            res[i] += jw
        return  ''.join(str(ch) for ch in res).lstrip('0')


# No.80  数字翻译成字符串 有多少种可能
def getTranslationCount(number):
    """
    从后往前翻译
    """
    def helper(s):
        # 一个数字至少有一种翻译，因此可以先设置一个全为1的数组，长度对应数字的位数加一
        counts = [1] * (len(s)+1)
        # 对于前面的n-1位
        for i in range(len(s)-2, -1, -1):
            count = counts[i+1]
            # 如果第i位和第i+1位可以组合成一个10-25的数字，那么g(i,i+1) = 1
            # f(i) = f(i+1) + g(i, i+1) x f(i+2)
            if 10 <= int(s[i:i+2]) <= 25: # 由于我们设置的数组长度是位数+1，因此这里i+2不可能越界
                count += counts[i+2]
            counts[i] = count
        return counts[0]

    if number < 0:
        return 0
    return helper(str(number))


# No.81
# 去除重复字母使得字典序最小
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        n = len(s)
        stack = [s[0]]
        for i in range(1,n):
            if s[i] not in stack and s[i]>stack[-1]:
                stack.append(s[i])
            elif s[i] not in stack:
                while stack and s[i]<stack[-1] and stack[-1] in s[i+1:]:
                    stack.pop()
                stack.append(s[i])
        return "".join(stack)




# No.82 正则表达式匹配
class Solution:
    def isMatch(self, s, p):
        m, n = len(s), len(p)

        def matches(i, j):
            if i == 0:
                return False
            if p[j - 1] == '.':
                return True
            return s[i - 1] == p[j - 1]

        f = [[False] * (n + 1) for _ in range(m + 1)]
        f[0][0] = True
        for i in range(m + 1):
            for j in range(1, n + 1):
                if p[j - 1] == '*':
                    f[i][j] |= f[i][j - 2]
                    if matches(i, j - 1):
                        f[i][j] |= f[i - 1][j]
                else:
                    if matches(i, j):
                        f[i][j] |= f[i - 1][j - 1]
        return f[m][n]




# No.83
# 复制带随机指针的链表
class Solution(object):
    def __init__(self):
        self.visitedHash = {}

    def copyRandomList(self, head):

        if head == None:
            return None

        if head in self.visitedHash:
            return self.visitedHash[head]

        node = Node(head.val, None, None)

        self.visitedHash[head] = node

        node.next = self.copyRandomList(head.next)
        node.random = self.copyRandomList(head.random)

        return node



# No.84
# 乘积小于k的连续子数组
class Solution(object):
    def numSubarrayProductLessThanK(self, nums, k):
        # 子数组是单一元素或连续子序列
        # so, 滑窗处理
        ll = len(nums)
        if k < 2 or not ll:
            return 0
        res = 0
        l = 0
        product = 1  # 乘积值
        for r in range(ll):
            product *= nums[r]
            # product太大? 剔除滑窗左边的值
            while product >= k:
                product /= nums[l]
                l +=1
            res += (r-l+1)  # l-r+1这个范围都是满足要求的，单个元素也好，若干个连续元素也好
        return res



# No.85
# 最大矩形(全1)
class Solution(object):
    def maximalRectangle(self, matrix):
        if not matrix or not len(matrix[0]):
            return 0
        m,n = len(matrix), len(matrix[0])
        # 计算每一行，以它为首行的，最大高度
        height = [0]*n
        res = 0
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == "0":
                    height[j] = 0  # 因为是以此行为首行,故'0'就是0
                else:
                    height[j] += 1
            res = max(res, self.maxwide(height))
        return res

    def maxwide(self, height):
        # height是统计了列上的高，现在还要看横向的宽
        height.append(0)
        stack = []  # 保存宽的左边界 越后面值越大
        res = 0
        for i in range(len(height)):
            while stack and height[i]<height[stack[-1]]:
                s = stack.pop()  # 弹出最后元素，越后值越大，i-x就越小，也即宽越小
                # s是作为宽的左边界，i是右边界
                # height[s] 代表高, (i-stack[-1]-1) if stack else i 代表宽
                res = max(res, height[s]*((i-stack[-1]-1) if stack else i))
                # i-stack[-1] -1 -1是因为stack.pop()弹出了最尾元素，相当于少减了一个1 【越后值越大！！！好好理解!!!】
            stack.append(i)
        return res



# No.86 全排列
class Solution(object):
    def permute(self, nums):
        # 回溯法
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i]+nums[i+1:], tmp+[nums[i]])
        backtrack(nums, [])
        return res
# 牛客ac版
# -*- coding:utf-8 -*-
class Solution:
    def __init__(self):
        self.res = []
    def Permutation(self, ss):
        if ss == '':
            return []
        ll = len(ss)
        # 为啥要把ss转成list?
        # 因为string是不可变的,list才可方便后续char换位
        s = []
        for i in range(ll):
            s.append(ss[i])
        self.PermutationHelper(s, 0, ll)
        self.res = list(set(self.res))
        self.res.sort()
        return self.res

    def PermutationHelper(self, ls, pre, end):
        if end <= 0:
            return
        if pre == end-1:
            # 指针走到最后一个元素了已经
            self.res.append(''.join(ls))  # ''.join(ls)把list ls转为str
            # 再append进res
        else:
            for i in range(pre, end):
                self.swap(ls, pre, i)
                self.PermutationHelper(ls, pre+1, end) # 递归
                self.swap(ls, pre, i)

    def swap(self, str_, i, j):
        str_[i], str_[j] = str_[j], str_[i]




# No.87
# 连续子数组的最大和
class Solution:
    def FindGreatestSumOfSubArray(self, array):
        if array == None:
            return
        p1 = 0  # 指向最大和子序列的首末元素
        p2 = 0  # 指向最终和最大的子序列的,首末元素
        maxsum = array[0]  # 初始化最大和值
        j = 0  # j用来计数遍历完整个list. 不能使用p2的指向作为list是否遍历完的判断. 因为p2是指向最大和子序列的末尾,它不一定值会等于list的尾.
        while j < len(array) and p2 < len(array):
            j += 1
            cursum = sum(array[p1:p2+1])
            if cursum > maxsum:
                maxsum = cursum  # 更新已经计算过的最大的子序列和
            if cursum > 0:  # 这里设置成>0,也就是当前面的子序列和=0时也把前面的给丢掉.
                p2 += 1
            else:  # 当前得到的和是负数,那么就摒弃之前的所有子序列.
                p1 = p2 + 1
                p2 = p1
        return maxsum

























































































