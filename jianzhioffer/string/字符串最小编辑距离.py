# coding=utf-8
# 字符串最小编辑距离  dp

# https://www.cnblogs.com/boris1221/p/9375047.html

# 状态转移方程，我们要得到dp[i][j]的值，假设s1[i-1]和s2[j-1]之前的都已经相等了，那么如果s1[i]==s2[j]，
# 显然不需要进行操作，dp[i][j]==dp[i-1][j-1]；如果s1[i]!=s2[j]，那么到达dp[i][j]的就有三条路，
# 分别从dp[i-1][j-1]、dp[i-1][j]、dp[i][j-1]，对应的含义分别是修改字符、删除字符和插入字符，


def eidt_1(s1, s2):
    len_str1 = len(s1) + 1
    len_str2 = len(s2) + 1

    matrix = [[0] * (len_str2) for i in range(len_str1)]

    for i in range(len_str1):
        for j in range(len_str2):
            if i == 0 and j == 0:
                matrix[i][j] = 0

            elif i == 0 and j > 0:
                matrix[0][j] = j
            elif i > 0 and j == 0:
                matrix[i][0] = i

            elif s1[i - 1] == s2[j - 1]:
                matrix[i][j] = min(matrix[i - 1][j - 1], matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
            else:
                matrix[i][j] = min(matrix[i - 1][j - 1] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j] + 1)
    return matrix[len_str1 - 1][len_str2 - 1]


if __name__ == '__main__':
    s1 = 'cafe'
    s2 = 'coffee'
    res = eidt_1(s1, s2)
    print(res)
