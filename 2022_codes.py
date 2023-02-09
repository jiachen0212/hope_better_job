# coding=utf-8
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# dp动态规划 # 
# 跳跃游戏, leetcode55
def canJump(nums):
    # 贪心法 or dp动态规划

    '''
    1. dp
    dp[i]: 0表示不可到达, 1表示可到达
    dp[0]一定==1, 
    0~lens-1, dp[i]==0, 则i起至后面均不可达, 直接False
    若i+nums[i]大于等于数组长, 则一定True: i+nums[i] >= lens-1
    剩下的, i+1~dp[i]+nums[i]之间, 肯定是可达的: range(i+1, i+nums[i]+1)
    因为可跳1~nums[i]步
    '''
    lens = len(nums)
    # dp = [0]*lens
    # dp[0] = 1
    # for i in range(lens):
    #     if dp[i] == 0:
    #         return False
    #     if i+nums[i] >= lens-1:
    #         return True
    #     for j in range(i+1, i+nums[i]+1):
    #         dp[j] = 1

    # return False 

    '''
    2. 贪心算法
    遍历数组一次, 持续更新可到达的max值
    最远可到达, max(mmax, i+nums[i]) 

    ''' 
    mmax = 0
    for i in range(lens):
        if i > mmax:
            return False
        mmax = mmax if mmax >= i+nums[i] else i+nums[i]

    return mmax >= lens-1

nums = [3,2,1,0,4]
print('跳跃游戏', nums, canJump(nums))


# leetcode72 编辑距离
def minDistance(word1, word2):
    # word1,2某一个为空, 则另一个直接删len(word)次         
    if not word1 and word2:
        return len(word2)
    if not word2 and word1:
        return len(word1)
    
    # 需注意的是这两行, n m顺序别混了..
    n, m = len(word1), len(word2)
    dp = [[0]*(m+1) for i in range(n+1)]

    # 横竖初始化
    dp[0] = [i for i in range(m+1)]
    for i in range(n+1):
        dp[i][0] = i 
    
    for i in range(n):
        for j in range(m):
            if word1[i] == word2[j]:
                dp[i+1][j+1] = dp[i][j]
            else:
                # 增删替, 操作都得+1 取min(dp)去做
                dp[i+1][j+1] = min(dp[i+1][j], dp[i][j+1], dp[i][j]) + 1

    return dp[-1][-1]

word1 = "horse"
word2 = "ros"
print('编辑距离:', word1, word2, minDistance(word2, word1))



# leetcode115 不同的子序列 
'''
给定一个字符串 S 和一个字符串 T，计算在 S 的子序列中 T 出现的个数。

一个字符串的一个子序列是指，通过删除一些（也可以不删除）字符且不干扰剩余字符相对位置所组成的新字符串。
（例如，"ACE" 是 "ABCDE" 的一个子序列，而 "AEC" 不是）
'''
def numDistinct(s, t):
    if not s:
        return 0

    # 更新dp[i][j]
    def dfs(i, j, s, t, dp):
        if j < 0:
            return 1
        if i < 0:
            return 0
        if dp[i][j] >= 0:
            return dp[i][j]
        temp_res = dfs(i-1, j, s, t, dp)
        # 当si==tj的时候, 可有两种做法: 
        # 1. 不用si, 继续在0~i-1之间找t(0,j). 也就是line103
        # 2. 用si, 则在0~i-1之间找t(0,j-1). 也就是line108
        if s[i] == t[j]:
            temp_res += dfs(i-1, j-1, s, t, dp)
        dp[i][j] = temp_res
        return dp[i][j]

    n, m = len(s), len(t)
    dp = [[-1]*m for _ in range(n)]
    # dp[i][j]: s(0~i)中有多少个t(0~j)
    return dfs(n-1, m-1, s, t, dp)

s = "babgbag"
t = "bag"
print('不用的子序列', s, t, numDistinct(s, t))

# dp写法
def numDistinct1(s, t):
    n, m = len(s), len(t)
    if n < m:
        return 0 
    dp = [[0]*m for _ in range(n)]
    # dp[i][j]表示s(0~i)里有多少t(0,j)
    dp[0][0] = 1 if s[0]==t[0] else 0
    for i in range(1, n):
        temp = 1 if s[i]==t[0] else 0
        dp[i][0] = dp[i-1][0] + temp
    for i in range(1, n):
        for j in range(1, m):
            dp[i][j] = dp[i-1][j]
            if s[i] == t[j]:
                # 和dfs写法一样, 如果st在ij位置处一致, 则用不用这个si有两种可能
                dp[i][j] += dp[i-1][j-1]
    return dp[-1][-1]
print(s, t, numDistinct1(s, t))

# dp优化空间, 只需要一维m
def numDistinct2(s, t):
    n, m = len(s), len(t)
    if n < m:
        return 0 
    dp = [0]*m
    if s[0]==t[0]:
        dp[0] = 1 
    for i in range(1, n):

    	# range(m-1,0,-1)完成一次t单词遍历
    	# t[0]没遍历到
        for j in range(m-1, 0, -1):
            if s[i] == t[j]:
                dp[j] += dp[j-1]
        if s[i] == t[0]:
            dp[0] += 1 
    return dp[-1]
print(s, t, numDistinct2(s, t)) 


# leetcode124 二叉树中的最大路径和
'''
任意节点出发到任意节点结束
'''
class Solution:
    def maxPathSum(self, root):
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


# leetcode174 地下城游戏
'''
从左上走到右下, 中途不可<0. 考虑右下往左上的dp实现
'''
def calculateMinimumHP(dungeon):
    n, m = len(dungeon), len(dungeon[0])
    dp = [[10000000000]*(m+1) for _ in range(n+1)]
    # 最后一步, 从左边or上边来, 值得至少为1
    dp[n][m-1]=dp[n-1][m] = 1
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            # 由下往上走
            mmin = min(dp[i][j+1], dp[i+1][j])
            dp[i][j] = max(1, mmin-dungeon[i][j])
    return dp[0][0]
dungeon = [[-2,-2,3],[-5,-10,1],[10,30,-5]]
print('地下城勇士: ', dungeon, calculateMinimumHP(dungeon))


# leetcode188 买股票最佳
# 一共可交易k次求最大利润 
def maxProfit(k, prices):
	# dp[0/1][i][j]: 0/1表示买入还是卖出, i为股票index, j为交易index
    n = len(prices)
    if not n:
        return 0
    if k >= n//2:
        # 可以一直买入卖出
        res = 0
        for i in range(1, n):
            res += max(0, prices[i]-prices[i-1])
    # dp0为最后一次操作是买入, dp1为最后一次操作是卖出
    # 买入+卖出等于1次交易(你必须在再次购买前出售掉之前的股票）
    dp0 = [-prices[0]]*(k+1)
    dp1 = [0]*(k+1)
    for p in prices[1:]:
        for j in range(1, k+1):
            # dp0[j]: 不买第i支股票
            # or j-1次的最后一下是卖出: dp1[j-1], 第j次是买入-p
            dp0[j] = max(dp0[j], dp1[j-1]-p)

            # dp1[j]:不卖第i支股票, 则: dp1[i-1][j]
            # or 卖第i支股, 则dp0[i-1][j]: dp0[i-1][j]+p
            # 注意不用j-1, 因为买入+卖出==一次交易
            dp1[j] = max(dp1[j], dp0[j]+p)
    # 最后肯定要清仓的, dp1.
    return max(dp1[k], 0)
k = 2
prices = [3,2,6,5,0,3]
print('股票最大收益: ', k, prices, maxProfit(k, prices))


# leetcode309 最佳股票 含冷冻期(依次交易后得隔一天才能再交易)
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
        dp1[i] = max(dp1[i-1], dp0[i-1]+prices[i])  # 本来这里不需要dp0[i-1]的,dp0[i]即可, 但这里有冷冻期, 故倒退一天.
    return dp1[n-1]
prices = [3,2,6,5,0,3]
print('冷冻一天, 股票最大收益: ', prices, maxProfit(prices))


# leetcode714 含手续费 股票最佳
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
prices, free = [1,3,7,5,10,3], 3 # [1, 3, 2, 8, 4, 9], 2
print('包含手续费的股票收益: ', prices, free, maxProfit(prices, free))


# leetcode198 打家劫舍1
# 相邻不能偷, 首尾不相连
def rob(nums):
    if not nums:
        return 0
    n = len(nums)
    if n<=1:
        return nums[0]
    # dp = [0]*n
    # dp[0]=nums[0]
    # dp[1]=max(nums[0], nums[1])
    # for i in range(2, n):
    #     # 在位置i, 偷不偷i, 更新i处的最大值
    #     dp[i] = max(dp[i-1], dp[i-2]+nums[i])
    # return dp[-1]

    # 空间优化: 每个位置i的更新只和i-1, i-2有关系, 故一直更新这两个变量就可以了.
    i_2, i_1 = 0, 0  # 分别代表i-2位置和i-1位置的最大偷窃值
    # 遍历数组维护这个最大偷窃结果
    res = 0
    for i in range(n):
        res = max(i_1, i_2+nums[i])
        # 更新i-1,i-2
        i_2 = i_1
        i_1 = res  # res就是i位置的最大值, 更新过程中复制给i_1
    return res 
nums = [2,7,9,3,1]
print('打家劫舍1: ', nums, rob(nums))

# leetcode213 打家劫舍2
# 首尾相连: 分两种情况, 偷第一家,则最后一家不可偷, (1~n-1)内找最大;
# 不偷第一家, 则(2~n)内找最大. 比较这两种谁更大, 即可.
def rob1(nums):
    if not nums:
        return 0
    n = len(nums)
    if n <= 2:
        return max(nums[0], nums[-1])

    def rob_(a, b, nums):
        i_1, i_2, res = 0,0,0
        for i in range(a, b):
            res = max(i_1, i_2+nums[i])
            i_2 = i_1
            i_1 = res 
        return res 
    res1 = rob_(0, n-1, nums)
    res2 = rob_(1, n, nums)
    return max(res2, res1)


# leetcode337 打家劫舍3
'''
房屋之间的连接是二叉树, 有连接的节点被偷会报警
'''
# 和上题一样, 上题的第一户有偷和不偷两种情况, 这题则是, 对于节点r, 偷r则不能偷r的左右子, 也是两种情况.
def rob(root):
	# dfs函数计算: 不偷root or 偷root, 分别可获得的金额 
	def dfs(root):
		if not root:
			return [0, 0]
		l_01 = dfs(root.left)
		r_01 = dfs(root.right)
		# 不偷root, 则可偷左子和右子, 取左右各自的最大, 得到root0
		root0 = max(l_01) + max(r_01)

		# 偷root, 为root1值
		root1 = l_01[0] + r_01[0] + root.val # 左边的不偷root+右边的不偷root, 加上本身root的val 

		return [root0, root1]

	res = dfs(root)
	return max(res)



# leetcode233 数字1的个数
'''
给定一个整数 n，计算所有小于等于 n 的非负整数中数字 1 出现的个数
n = 13, : 1, 10, 11, 12, 13, 6个1
'''
# 数学方法处理, 每一位出现的1, 相加.
def countDigitOne(n):
	count = 0
	l = n 
	res = 0
	while l:
		l //= 10
		count += 1
	# m是和n位数一致的10的次方值
	m = 10 ** (count-1)
	while m:
		div = n // m  # 最高位数值
		mod = n % m 
		if div % 10 == 1:
			res += div//10*m + mod+1
		elif div % 10 == 0:
			res += div//10*m 
		else:
			res += (div//10+1)*m   
		m //= 10 
	return res 
print(countDigitOne(13), '个1')


# leetcode312 戳气球
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


# leetcode354 俄罗斯套娃信封问题
'''
信封的宽高都大于另一个, 则可套信封成功. 因为有hw两个维度, 故先对h升序排序, 保证h上排后的可以装下排前的.
另外还需要后面的w大于前面的w(才能装下), 故问题转化为: 在h升序排序后, 找w的最长升序长度, 即为可套的总信封个数.
'''
def maxEnvelopes(envelopes):
	# dp[len]表示最长升序列长度为len,的最后一个元素的最小值. (这个值越小,后面可接入的元素就可越多)
	'''
	问题转化为, 新来一个元素a[i], if a[i] > dp[len], 则len+1, dp[len+1]更新为a[i]
	if a[i]<dp[len], 则需要len-1~0往前遍历, 把a[i]的值更新给对应的dp[x]
	so, 问题变为把数组的每个元素插入dp对应的位置, 可用二分法完成这件事, 时复O(nlogn)
	''' 
	# 信封(h,w), 先对h最升序排序, 
	envelopes.sort(key=lambda x: (x[0], -x[1]))
	
	# 接着找w的最长升系列长度. 也就是nums的最长升序长度
	nums = [e[1] for e in envelopes]
	n = len(nums)
	# 初始化dp, dp的len就是最终res
	dp = []
	# 把nums的每个w值插入dp的合适位置.
	# dp[len]为最长升序长度为len的, 最末(大)元素的最小值
	from bisect import bisect_left
	for w in nums:
		# bisect_left(ls, x)为x值可插入ls的最早index, bisect_right为最晚index. [当ls中有很多重复元素==x, left和right的返回值会有区别]
		# 直接把w插入dp中, 且每次插入选最靠前的位置. 
		idx = bisect_left(dp, w)
		# 刚好等于len(dp), 则刚好append到最后一位.
		if idx == len(dp):
			dp.append(w)
		else:
			# idx小于len(dp), [记住是不可能出现idx大于len(dp)的]
			# 把dp的idx位置的值更新为w, 这个w值是比之前dp[idx]更小的值
			dp[idx] = w
	return len(dp)	
envelopes = [[5,4],[6,4],[6,7],[2,3]]
print('俄罗斯套信封: ', envelopes, maxEnvelopes(envelopes))


# leetcode376 摆动序列
'''
相邻元素之差保持: 正负正负... 则为摆动序列
贪心算法做
'''
def wiggleMaxLength(nums):
	n = len(nums)
	if n <= 1:
		return n 
	# 默认是先升后降, so初始化的ord为-1先
	pre_ord = -1
	res = 1
	for i in range(1, n):
		if nums[i] == nums[i-1]:
			continue
		# i与i-1的大小关系, i>i-1则1, 否则0
		cur_ord = 1 if nums[i] > nums[i-1] else 0 
		if cur_ord != pre_ord:
			# 出现+1-1摆动
			res += 1
		# 更新pre_ord状态(每一对i-1和i都要重新更新pre_ord)
		pre_ord = cur_ord
	return res 
nums = [1,2,3,4,5,6,7,8,9]# [1,17,5,10,13,15,10,5,16,8] # [1,7,4,9,2,5]
print('摆动序列长度: ', nums, wiggleMaxLength(nums))


# leetcode390 消除游戏
# 是一道数学归纳题的感觉: n=2k, 则满足: f(2k)=2(k+1-f(k)); 当n=2k+1, 也一样第一轮就会把最末的那个2k+1消除了.
# f(2k+1) = 2(k+1-f(k))
def lastRemaining(n):
        return 1 if n==1 else 2*(n//2+1-lastRemaining(n//2))
n = 9
print('正反向间隔1个删除: ', n, lastRemaining(n))


# leetcode689 三个无重叠数组的最大和
def maxSumOfThreeSubarrays(nums, k):
	'''
	dp[i][j]: (0,i)内组成了j对子数组, dp[i][j]为子数组和的最大值. j<=3
	分取nums[i]和不取nums[i]两种情况, 分别来更新dp[i][j]
	1. 取nums[i], 则dp[i][j] = dp[i-k][j-1]+nums[i+1-k:i] k是子数组的长度
	2. 不取nums[i], 则dp[i][j] = dp[i-1][j]
	path[i][j]存储的则是dp[i][j]最大值时候的, 每段子序列的末尾index
	'''
	n = len(nums)
	ssum = [0]*n
	first_sum = 0
	for i in range(k):
		first_sum += nums[i]
		ssum[i] = 0
	# (0,k)内的和初始化给k位置处
	ssum[k-1] = first_sum
	# 从k开始, ssum的每个index更新. 
	for i in range(k, n):
		# 移动到i, 取i的话, 则ssum的变化值: nums[i]-nums[i-k]. 这个好好理解下.
		first_sum += nums[i]-nums[i-k]
		ssum[i] = first_sum
	# 拆分为3组, 故N=3, 
	N = 3
	dp = [[0]*(N+1) for i in range(n)]
	path = [[0]*(N+1) for i in range(n)]
	# 到k-1为止, 只进行了1次分组的, 故dp[k-1][1], 和为上面init的ssum[k-1]
	dp[k-1][1] = ssum[k-1]
	path[k-1][1] = k-1 
	for i in range(k, n):
		# j可取: 1,2,3
		for j in range(1, N+1):
			# 不取nums[i]时的dp, path更新
			dp[i][j] = dp[i-1][j]
			path[i][j] = path[i-1][j]
			# 取ssum
			if dp[i][j] < dp[i-k][j-1] + ssum[i]:
				dp[i][j] = dp[i-k][j-1] + ssum[i]
				path[i][j] = i
	# 更新完了dp,path, 则取出每个分段的起点index
	res = []
	# 最后一个值
	idx = path[n-1][N]
	# 最后一组的最末元素index==idx, idx-(k-1)则为最末组的起点index
	res.append(idx-k+1)
	# 逆序取出每一段的最末元素
	for i in range(N-1, 0, -1):
		idx = path[idx-k][i]
		res.append(idx-k+1)
	return res[::-1]
nums = [1,2,1,2,1,2,1,2,1] # [1,2,1,2,6,7,5,1]
k = 2
print('拆分三个子数组, 得到和最大, 三段的起始index: ', nums, k, maxSumOfThreeSubarrays(nums, k))


# leetcode907 子数组的最小值之和




# 四面体方案个数 (三菱柱, S顶点, ABC三个点且对称)
'''
从S出发, 每次任意选一条棱走到另一个顶点, 可重复走过所有顶点和棱.
问走k次之后回到S的方案数是多少?

dp[i][0]走k次后回到S; dp[i][1]走k次后回到(ABC中的某一点, ABC三个点是对称的, so都一样.)

i了在S点, 是i-1在(ABC)三处之和: dp[i][0] = dp[i-1][1]*3
i了在A(orBC一样的), 是i-1在BC, 和i-1在S之和: dp[i][1] = dp[i-1][1]*2 + dp[i-1][0]

'''
def simianti(k):
	mod = 1e9+1
	# 起点在S故为1, dp1初始值则为0
	dp0, dp1 = 1, 0
	for i in range(1, k+1):
		# %mod是保证可以取到够大的数值 
		s = (dp1*3)%mod
		abc = (dp1*2 + dp0)%mod
		dp0, dp1 = s, abc 
	return dp0
k = 10
print('四面体走k步, 回到S点的可能: ', simianti(k))


# leetcode1186 选出一个连续子数组, 在内删除一个数(但删除后子数组不可为空,也可不删), 使子数组的和最大
'''
解决思路: 类似求连续子数组的最大和, 也就是本题不删除元素的情况. 
如果删除, 则需要把子数组的删除元素的左右两边, 分为俩子子连续数组. 持续更新删除i后的右子数组即可.
联系子数组最大和: 状态方程: dp[i]为以i结尾的最大和连续子数组. dp[i] = max(dp[i-1], 0) + arr[i]

'''
def maximumSum(arr):
	n = len(arr)
	dp = [arr[0]]*n 

	# 连续子数组的最大和
	for i in range(1, n):
		dp[i] = max(dp[i-1], 0) + arr[i]

	# 看看要不要删除子数组中的某个元素
	# res初始值是上面不删除元素情况下的最大值
	# sub_right为删除i元素的话, 其右边的子数组的最大值
	res, sub_right = max(dp), arr[-1]  
	# 1~n-2反向遍历
	for i in range(n-2, 0, -1):
		res = max(res, dp[i-1]+sub_right)
		sub_right = max(sub_right, 0) + arr[i] 
	return res 
arr = [-1,-1,-1] # [1,-2,-2,3] # [1,-2,0,3]
print('删除or不删除某个元素, 子数组最大和: ', arr, maximumSum(arr))


# leetcode 
'''
币值为25分、10分、5分和1分，编写代码计算n分有几种表示法.
'''
def waysToChange(n):
	pass 



# leetcode 三步问题. 上n阶台阶, 每次可走1 2 3 步, 求可能得方法.
# f(i) = f(i-1)+f(i-2)+f(i-3)
def waysToStep(n):
    f = [1,2,4] # 有123层台阶时候可走的方案数. 2 = 1+1or2; 3 = 1+1+1 or 1+2 or 2+1 or 3
    for i in range(3, n):
        f[i%3] = f[(i-3)%3] + f[(i-2)%3] + f[(i-1)%3]
        f[i%3] %= 1e9+7
    return int(f[(n-1)%3])
n = 5   # 做%取余处理, 是为防止n太大.
print('跳台阶三步问题: ', n, waysToStep(n))


# leetcode62 圆圈中最后剩下的数
# 从0开始, 每次删除第m个数, 求最后剩下的数
'''
(x+k+1)%n = (x+m)%n
f(n) = (f(n-1)+m)%n
'''
def lastRemaining(n, m):
	last = 0
	for i in range(2, n+1):
		last += m 
		last %= i 
	return last
n, m = 5, 3
print('圆圈数0~n-1, 每次删第m个数, 最后剩下: ', n, m, lastRemaining(n, m))

# leetcode46 数字转字符串
# 0~25 -> a~z  
# 所以给一串数字, 求其可转位字符串的种数
# 123: bcd md bx

'''
动态规划: 
1. nums[i-1, i]两位在10~25的话, 则可: 表示一个字母 or 俩单独的字母
	dp[i]表示前i个数字可表示的字符串种数
	dp[i] = dp[i-1] + dp[i-2]. 也就是从最后两位i-2,i-1考虑, dp[i-2]表示最后两位用来表示一个字母, 
	dp[i-1]则表示最后一位单独作为一个字母, 则i-2位是单独还是和i-3合并, 无需管

2. nums[i-1, i]<10的话, 则直接: dp[i] = dp[i-1]

优化方法可写为:f(n), 最后两位值是: last=n%100, 
if last: 10~25, 则f(n)=f(n/10)+f(n/100);
else: f(n) = f(n/10)
if n < 10了, 则return 1 

'''
def translateNum(num):
    if num < 10:
        return 1
    last = num % 100
    if 10 <= last <= 25:
        return translateNum(num//100) + translateNum(num//10) 
    else:
        return translateNum(num//10)
num = 123 # 258
print('数字可组成字符串的种数: ', num, translateNum(num))

