#coding=utf-8
# 求n个骰子的点数和及对应的出现概率..
# 思路: F(n,s) = F(n-1,s-1)+F(n-1,s-2)+F(n-1,s-3)+F(n-1,s-4)+F(n-1,s-5)+F(n-1,s-6)

############################# 方法一 ###################################
# n为骰子的个数
def dp_probability(n, s, dmax = 6, dmin = 1):
    if s < n * dmin or s > n * dmax:
        return 0
    dp1 = [0] * (n * dmax + 1)  # 第一轮骰子的和情况

    # init dp[1, :]
    for i in range(1, dmax + 1):
        dp1[i] = 1   # [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        # dp1的index表示是出现的点数:1~6,元素值表示该点数会出现的次数. 所以把各个位上的元素值初始化为1没毛病..
    # print dp1, 'dp1 ...1'

    for i in range(2, n + 1):  # i: 骰子的编号
        dp2 = [0] * (n * dmax + 1)  # 第二轮骰子的和情况
        for j in range(dmin * i, dmax * i + 1):  # j: range of i dices
                      # 当是第二个骰子,即i=2,j:2~12
            for k in range(dmin, dmax + 1):  # k: range of new added dice  1~6
                if j > k:
                    dp2[j] += dp1[j - k]  # F(n,s) = F(n-1,s-1)+F(n-1,s-2) F(n-1,s-3)+F(n-1,s-4)+F(n-1,s-5)+F(n-1,s-6)
                    # 即dp2[j]等于dp1中的j-1~j-6相加
        print dp2, '####'
        dp1 = dp2
        # dp2表示每个可能出现的点数和,的出现次数. index表示点数和,对应位置上的元素值表示出现的次数.
    print "total = {0}, prob = {1}%".format(dp2[s], dp2[s] * 100 / dmax ** n)
    return dp2[s]

dp_probability(2, 11)



######################### 方法二 ###################################
def setTo1(dices, start, end):
    for i in range(start, end):
		dices[i] = 1

def probability(n, s, dmax = 6, dmin = 1):
	if s < n * dmin or s > n * dmax: return 0
	dices = [1] * n  # 首先使得每个骰子的点数均是1,即达到最小和
	i = n - 1
	total = 0

	while i >= 0:
		curSum = sum(dices)
		if curSum == s:
			print dices
			total += 1
			# find first one that can +1
			for j in range(i, -1, -1): # i 代表现有的骰子数目
				if dices[j] < dmax and s - sum(dices[0:j+1]) >= n - j*dmin:
					dices[j] += 1  # 把骰子j的数字加1
					setTo1(dices, j + 1, n)
					i = n - 1
					break
				else:
					i -= 1
		elif curSum < s:
			if dices[i] < dmax:
				dices[i] += 1
				i = n - 1
			else:
				i -= 1

	print "total = {0}, prob = {1}%".format(total, total*100/dmax**n)
	return total

probability(2, 12)