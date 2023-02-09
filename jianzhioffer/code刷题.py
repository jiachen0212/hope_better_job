# coding=utf-8
# python获取用户输入



'''
版本1

# 这三行的作用是：每一行输入一个数字，然后最后一个数字输完后ctrl+D结束，会把以上输入的
# 数字组合成一个数组

import sys
values = sys.stdin.readlines()[:]  # 从输入的第几行开始放入数组，可以在[]内设置
# 如 [1:]表示输入的第一个数不要，从第二个输入的开始放数字进数组
values = [int(v.strip()) for v in values]
print(values)

'''



'''
版本2

import sys
try:
    while True:
        line1 = sys.stdin.readline().strip()
        # if line1 == '':
        #     break
        line2 = sys.stdin.readline().strip()
        l = list(map(int, line1.split()))
        a = [int(n) for n in line1.split()]

        l = list(map(int, line2.split()))
        b = [int(n) for n in line2.split()]
        # 一共是两行输入，a是一行  b是一行; 这样就得到 a b 数组了


        # 这里执行 a b 相关的函数然后：
        # print(res)


        break   # 这里break出来就好了
except:
    pass

'''




'''
# 拼多多code1  a是只有一个位置不满足递增的数组，请在b里找一个b中最大的替换放入使a递增、
# 找不到的话 返回'No'
# a = [2, 5, 8, 7, 15]
# b = [1, 8, 9, 14, 12]

# coding=utf-8
import sys
try:
    while True:
        line1 = sys.stdin.readline().strip()
        line2 = sys.stdin.readline().strip()
        l = list(map(int, line1.split()))
        a = [int(n) for n in line1.split()]

        l = list(map(int, line2.split()))
        b = [int(n) for n in line2.split()]

        # print(a, 'a')
        # print(b, 'b')

        la = len(a)
        falg = 1
        for i in range(1, la):
            if a[i] < a[i-1]:
                if i == la - 1:
                    tmp1 = a[-2]   # 最末那个不符合增序，只需要把倒数第二个数记下
                    # 找一个大于tmp1的就ok
                else:
                    tmp1, tmp2 = a[i-1], a[i+1]
                    falg = 2
        b = list(set(b))
        b.sort()
        if falg == 1:
            if b[-1] < tmp1:
                print 'No'
                break
            else:
                print b[-1]
                break
        else:   # 也就是需要比较tmp1和tmp2
            if b[0] > tmp2 or b[-1] < tmp1:
                print 'No'
                break
            else:
                ca1 = []
                ca2 = []
                ca3 = []
                for j in range(len(b)):
                    if b[j] > tmp1:
                        ca1.append(b[j])
                    if b[j] <= tmp2:
                        ca2.append(b[j])
                for r in range(len(ca1)):
                    if ca1[r] in ca2:
                        ca3.append(ca1[r])
                if not ca3:
                    print 'No'
                    break
                else:
                    print ca3[-1]
                    break
except:
    pass

'''



'''
# 流利说code  和最大连续数组
# 输入的第一个数是数组的元素个数，第二个开始才是数组元素
import sys

values = sys.stdin.readlines()[1:]
values = [int(v.strip()) for v in values]
dp = []

for i,v in enumerate(values):
    if(i==0):
        dp.append(v)
    else:
        if(v+dp[-1]>v):
            dp.append(v+dp[-1])
            pass
        else:
            dp.append(v)
print(max(dp))

'''




# https://www.nowcoder.com/test/question/340e0d941b614a12a661d8d0285decf3?pid=17800753&tid=25472545
# 以下是深信服的6道编程题

'''
1. 子串模糊匹配：  ac了
exam：
abcdefg
a?c  3
aabcddefg
a?c  4
aabcddefg
a?d  5
aabcddefg
b?e  这种第一个没配上的，就会返回-1
有？直接count+=3


# coding=utf-8
import sys
try:
    while True:
        # 这两行可直接得到strings
        s1 = sys.stdin.readline().strip()
        s2 = sys.stdin.readline().strip()
        ans = -1
        if not s1 or not s2:
            print 0
            break
        if len(s1) < len(s2):
            print ans
            break
        i,j,count = 0,0,0
        while (i < len(s1) and j < len(s2)):
            if s1[i] == s2[j]:
                count = 0
                i += 1
                j += 1
            elif '?' == s2[j]:
                count += 3
                j += 1
            elif j-1>=0 and '?' == s2[j-1] and s2[j] != s1[i] and count:
                i += 1
                count -= 1
            else:
                break
        if j >= len(s2):
            ans = i
        print ans
        break

except:
    pass

'''



'''
ac 了
2.有K种颜色的小球(K<=10)，每种小球有若干个，总数小于100个。
现在有一个小盒子，能放N个小球(N<=8)，现在要从这些小球里挑出N个小球，放满盒子。
想知道有哪些挑选方式。注：每种颜色的小球之间没有差别。

请按数字递增顺序输出挑选小球的所有方式。

如有3种颜色，每种颜色小球的个数分别为a:1,b:2,c:3，挑出3个小球的挑法有：
003,012,021,102,111,120

输入：
第一行两个数字K N，分别表示小球种类数目和挑选的小球个数
第二行开始为每种小球的数目，共N行数据：
3 3
1
2
3
输出：
003
012
021
102
111
120
'''
# 递归

'''
# coding=utf-8
import sys
a = [0]*11
ans = [0]*11
k,n=0,0

def Print(k, ans):
    res = ''
    for i in range(k):
        res += str(ans[i])
    print(res)

def fun(i, n): # i可以认为是球类比的编号，ans[i]就是结果中，i类球放几个
    if n==0:  # n=0就是递归结束了，要放的球都安排好了～
        Print(k, ans)
        return
    if n<0 or k==i:
        return
    for j in range(a[i]+1):  # a[i]+1就是j可以取0～a[i]
    # 即j表示这类球放几个
        ans[i] = j
        fun(i+1, n-j)  # i+1是移动下，考虑放下一类球了～
    ans[i] = 0  # 没遍历到的ans位就是放0个球

try:
    while True:
        line1 = sys.stdin.readline().strip()
        l = list(map(int, line1.split()))
        k = l[0]  # 球共所少类
        n = l[-1] # 共需要n个球
        nums = sys.stdin.readlines()
        # a数组，index为球的类，index上的value为这类球有的数量
        a = [int(v.strip()) for v in nums]
        fun(0, n)
except:
    pass
'''



'''
ac 了
3. 把几段数据段合并
'''

'''
# coding=utf-8
# ac 70%
# 内存超了
import sys
lines = []
try:
    while True:
        values = sys.stdin.readlines()
        values = [list(map(int, v.split())) for v in values]

        alls = []
        for i in range(len(values)):
            for j in range(values[i][0], values[i][-1]):
                alls.append(j)
        res = list(set(alls))

        duns = []
        p1, p2 = 0, 0
        while p2 <= len(res) - 1:
            if res[p2] - res[p1] == p2-p1:
                if p1 == len(res)-1:
                    duns.append([res[p1], res[p1]+1])
                if p1 != p2 and p2 == len(res) -1:
                    duns.append([res[p1], res[p2]+1])
                p2 += 1
            else:
                duns.append([res[p1], res[p2-1]+1])
                p1 = p2

        for ans in duns:
            if len(ans)>1:
                print ans[0], ans[1]
            else:
                print ans[0]
        break
except:
    pass
'''



# import sys
# def getCo(n):
#     if(n%3 == 2):
#         return (n/3)*2+1
#     else:
#         return (n/3)*2
# while True:
#     line = sys.stdin.readline().strip()
#     if(line == ''):
#         break
#     l,r = [int(x) for x in line.split()]
#     print getCo(r)-getCo(l-1)




# coding=utf-8
# import sys
# try:
#     while True:
#         # 这两行可直接得到strings
#         n = sys.stdin.readline().strip()
#         nums = sys.stdin.readline().strip()
#         tmp = ''
#         for i in range(1, int(n)+1):
#             tmp += str(i)

#         s = ''
#         for i in nums:
#             if i != ' ':
#                 s += i

#         def perm(s=''):
#             if len(s)<=1:
#                 return [s]
#             res=[]
#             for i in range(len(s)):
#                 for j in perm(s[0:i]+s[i+1:]):
#                     res.append(s[i]+j)
#             return res

#         nn = perm(tmp)
#         ind = nn.index(s)
#         ans = nn[len(nn)-1-ind]

#         for i in ans:
#             print int(i),

#         break

# except:
#     pass


# 笔试ac
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass
# height = []
# for s in ss:
#     height.extend(s.split(','))
# his = []
# for num in height:
#     if num == '':
#         continue
#     his.append(int(num))

# def maxArea(height):
#         i = 0
#         j = len(height)-1
#         water = 0
#         while i<j:
#             water = max(water,(j-i)*min(height[i],height[j]))
#             if height[i]<height[j]:
#                 i+=1
#             else:
#                 j-=1
#         return water
# print maxArea(his)



# coding=utf-8
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num.split(','))
# except:
#     pass
# M = int(ss[-1][-1])
# a = []
# for i in ss[0]:
#     a.append(int(i))

# def coinChange(coins, amount):
#     dp = [amount+100] * (amount+1)  # 这里先把最坏的兑换可能预设好
#     dp[0]=0
#     for i in range(1, amount+1):
#         for coin in coins:
#             if i >= coin:
#                 dp[i] = min(dp[i], dp[i-coin]+1)
#     return dp[-1] if dp[-1] != amount+100 else -1

# res = coinChange(a,M)
# print res



# coding=utf-8
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass
# height = []
# for s in ss:
#     height.extend(s.split(','))
# his = []
# for num in height:
#     if num == '':
#         continue
#     his.append(int(num))

# def jump(nums):
#         l = len(nums)
#         if l==1: return 0

#         from collections import deque
#         q = deque()
#         res = 0
#         visited = [False for i in range(l)]
#         q.append(0)
#         visited[0] = True
#         while q:
#             for j in range(len(q)):
#                 node = q.popleft()
#                 for i in range(nums[node], 0, -1): #从最大开始找有助于加快速度
#                     new_index = node+i
#                     if new_index >= l-1:
#                         return res + 1
#                     if not visited[new_index]:
#                         visited[new_index] = True
#                         q.append(new_index)
#             res += 1
# res = jump(his)
# print res



# s =raw_input()
# s = s.split(' ')
# def fanzhuankeys(s):
#     s = list(s)  # 先把str变成list,方便使用reverse()  reverse()函数没有返回值,直接是逆序s的作用..
#     s.reverse()
#     s = ''.join(s)
#     news = ''
#     for key in s.split(' '):  # key就是每一个的单词了..
#         if key == '':
#             continue
#         # key = list(key)
#         # key.reverse()
#         key = ''.join(key)
#         news += key + ' '
#         # news += ' '
#     return news

# print fanzhuankeys(s)


# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass

# def rob(nums):
#         last = 0
#         now = 0
#         res = []
#         for i in nums:
#             last, now = now, max(last + i, now)
#         return now

# nums = ss[1:][0].split(' ')
# nn = []
# for n in nums:
#     nn.append(int(n))
# a= rob(nn)

# def shortestSubarray(A, K):
#         minLin = len(A) + 1
#         presum = [0]*minLin
#         for i in range(minLin-1):
#             presum[i+1] = presum[i] + A[i]

#         queue = []  # 存放连续子序列的index
#         for i in range(len(A)+1):   # i肯定是比当前的queue中的所有index都大的
#             while queue and presum[i] <= presum[queue[-1]]:
#                 queue.pop()  # 前面出现负,把前面的都依次pop掉
#             while queue and presum[i] - presum[queue[0]] >= K:
#                 res = i - queue[0]
#                 minLin = res if res < minLin else minLin
#                 queue.pop(0)  # 把更早的一些可以删除，使得子序列最短
#             queue.append(i)
#         return minLin if minLin < len(A)+1 else -1

# b = shortestSubarray(nn,a)
# print a,b




# https://blog.csdn.net/qq_34342154/article/details/77187929
# # 字符串解码
# def num(str1):
#     if str1 == None or str1 == "":
#         return 0
#     cur = 1 if str1[-1] != '0' else 0
#     nex = 1
#     for i in range(len(str1)-2, -1, -1):
#         if str1[i] == '0':
#             nex = cur
#             cur = 0
#         else:
#             tmp = cur
#             if int(str1[i]) * 10 + int(str1[i+1]) < 27:
#                 cur += nex
#             nex = tmp
#     return cur

# s = raw_input()
# print num(s)



# import numpy as np
# input = np.array([[40,24,135],[200,239,238],[90,34,94]])
# kernel = np.array([[0.0,0.6],[0.1,0.3]])

# def my_conv(input,kernel):
#     output_size = (len(input)-len(kernel)+1)
#     res = [[0]*output_size for i in range(output_size)]
#     for i in range(len(res)):
#         for j in range(len(res)):
#             res[i][j] = int(compute_conv(input,kernel,i,j))
#     return res

# def compute_conv(input,kernel,i,j):
#     res = 0
#     for kk in range(len(kernel)):
#         for k in range(len(kernel[0])):
#             res +=int(input[i+kk][j+k]) * float(kernel[kk][k])
#     return res

# # coding=utf-8
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass

# m = int(ss[0][0])
# imgs = ss[1:m+1]
# Input = []
# for hang in imgs:
#     Input.append(hang.split(' '))
# # print Input
# k_size = int(ss[m+1][0])
# kk = ss[-k_size:]
# kernel = []
# for k in kk:
#     kernel.append(k.split(' '))
# # print kernel
# res = my_conv(Input,kernel)
# for hang in res:
#     for n in hang:
#         print int(n),
#     print


# ss =raw_input()
# s = ss.split(' ')
# for ch in s:
#     if len(ch) % 2 == 1:
#         print ch[::-1],
#     else:
#         print ch,





# nums = raw_input()
# ns = nums.split(' ')
# def zhishu(n):
#     for i in range(2, n):
#         if n % i == 0:
#             return 0
#     return n

# tens = []
# ones = []
# for i in range(int(ns[0]), int(ns[1])):
#     res = zhishu(i)
#     if res != 0:
#         if i >= 10:
#             tens.append(int(str(i)[-2]))
#         ones.append(int(str(i)[-1]))
# print sum(tens) if sum(tens) < sum(ones) else sum(ones)





# coding=utf-8
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass
# first = ss[0]
# persons = ss[-int(ss[1]):]
# res = set()
# res.add(first)
# def checkcommon(res, qun):
#     for q in qun:
#         if q in res:
#             return True
#     return False

# for qun in persons:
#     q = qun.split(',')
#     flag = checkcommon(res, q)
#     if flag:
#         for name in q:
#             res.add(name.strip(' '))
# print len(list(res))

# nums = raw_input()
# ns = nums.split(' ')
# res = []
# for n in ns[1:]:
#     if n == 'A':
#         res.extend([12, 34])
#     elif n == 'B':
#         res.extend(['AB', 'CD'])
#     else:
#         res.append(int(n))
# print len(res)+1,
# for i in res:
#     print i,

# nums = raw_input()
# ns = nums.split(',')
# ns.sort()
# res = ''
# for n in ns:
#     res += n
#     res += ','
# print res[:-2]


# coding=utf-8
# class Solution(object):

#     def DFS(self, M, i, j, m, n):
#         if (i<0 or j<0 or i>=m or j>=n):
#             return 0
#         if M[i][j] == '#':
#             return 0
#         a = self.DFS(M, i-1, j, m, n)
#         b = self.DFS(M, i, j-1, m, n)
#         c = self.DFS(M, i+1, j, m, n)
#         d = self.DFS(M, i, j+1, m, n)
#         return a or b or c or d

#     def fun(self, M):
#         m, n = len(M), len(M[0])
#         visited = [[0]*n for i in range(m)]
#         change_s = 0
#         count = 0
#         for i in range(m):
#             for j in range(n):
#                 if M[i][j] == '#' and not change_s:
#                     visited[i][j] = '.'
#                     change_s = 1
#                 if M[i][j] == '.' or M[i][j] == 'S' and not visited[i][j]:
#                     res = self.DFS(M, i, j, m, n)
#                     count += res
#         return count

# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass
# ss = ss[1:]
# tmps = []

# for hang in ss:
#     if  hang[0].isdigit():
#         tmps.append(int(hang[0]))
# be = 1

# def ffun(s):
#     ss = []
#     for hang in s:
#         cache = []
#         # print hang, '==='
#         for ch in hang:
#             cache.append(ch)
#         ss.append(cache)
#     return ss

# for i in range(len(tmps)):
#     bins = ss[be: be+tmps[i]]
#     aa = ffun(bins)
#     be += be+tmps[i]
#     s = Solution()
#     res = s.fun(aa)
#     print 'Yes' if res == len(aa) else 'No'




# # coding=utf-8
# import sys
# ss = []
# try:
#     while True:
#         num =raw_input()
#         ss.append(num)
# except:
#     pass
# n = int(ss[0])
# nums = ss[-n:]
# labels = []
# scores = []
# for pair in nums:
#     labels.append(int(pair.split(' ')[0][0]))
#     scores.append(float(pair.split(' ')[1]))


# def AUC(label, pre):
#     pos = [i for i in range(len(label)) if label[i] == 1]
#     neg = [i for i in range(len(label)) if label[i] == 0]

#     auc = 0
#     for i in pos:
#         for j in neg:
#             if pre[i] > pre[j]:
#                 auc += 1
#             elif pre[i] == pre[j]:
#                 auc += 0.5

#     return float(auc) / (len(pos)*len(neg))

# print AUC(labels, scores)

