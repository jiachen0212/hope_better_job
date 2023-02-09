# coding=utf-8
# 最长公共前缀
def longestCommonPrefix(strs):
        s = ''
        for i in zip(*strs): # 这个 zip(*strs)简直太牛逼了...
        # 这个代码写的是真的完美，zip会在其中一个str遍历完后就跳出循环不再管剩下的了...
            # print(i)
            '''
            ('f', 'f', 'f')
            ('l', 'l', 'l')
            ('o', 'o', 'i')
            '''
            if len(set(i)) != 1:   # 这个set是真的很牛逼！！！
                return s   # 并非3个str都有这个char? 好了那你可以退出了...
            else:  # set(i)==1,表示三个str中都有这个字符
                s += i[0]
        return s


strs = ["flower","flow","flight"]
strs = ["dog","racecar","car"]
res = longestCommonPrefix(strs)
print(res)


# 动态规划做这道题：https://blog.csdn.net/wangdd_199326/article/details/76464333
# 求最长公共子序列 非连续的....
def longestsubcommstrs(s1, s2, l1, l2):
    c = [[0] * (l2+1) for i in range(l1 + 1)]  # c[i1][i2]
    b = [[0] * (l2+1) for i in range(l1 + 1)]  # 用来保存是选：
    # s10~s1m-2, s20~s2n-1 还是 s10~s1m-1, s20~s2n-2
    # b[][]就是一个标识符的作用
    for i in range(1, l1 + 1):   # 加1时为了c的index好计数
        for j in range(1, l2 + 1):
            if s1[i-1] == s2[j-1]:
                c[i][j] = c[i-1][j-1] + 1
                b[i][j] = 0
            elif c[i-1][j] > c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i][j] = 1
            else:
                c[i][j] = c[i][j-1]
                b[i][j] = -1
    return c, b


def  printlcs(b, s1, i, j):
    if i==0 or j == 0:
        return
    if b[i][j] == 0:
        printlcs(b, s1, i-1, j-1)
        print(s1[i-1], '==')
    elif b[i][j] == 1:
        printlcs(b, s1, i-1, j)
    else:
        printlcs(b, s1, i, j-1)


def main(s1, s2):
    l1 = len(s1)
    l2 = len(s2)
    c, b = longestsubcommstrs(s1, s2, l1, l2)
    printlcs(b, s1, l1, l2)


main('abcfbc', 'abfcab')





# leetcode 1143 最长公共子序列(非连续但char有先后顺序 return长度)
class Solution(object):
    def longestCommonSubsequence(self, text1, text2):
        l1, l2 = len(text1), len(text2)
        dp = [[0]*(l2+1) for i in range(l1+1)]
        for i in range(1, l1+1):
            for j in range(1, l2+1):
                if text1[i-1] == text2[j-1]:
                    dp[i][j] = dp[i-1][j-1]+1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[-1][-1]


