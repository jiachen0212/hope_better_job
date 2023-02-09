# coding=utf-8
# 字符串相乘

'''
num1的第i位(高位从0开始)和num2的第j位相乘的结果在乘积中的位置是[i+j, i+j+1]
这个规律是解题关键

例: 123 * 45,  123的第1位 2 和45的第0位 4 乘积 08 存放在结果的第[1, 2]位中
  index:    0 1 2 3 4

                1 2 3
            *     4 5
            ---------
                  1 5
                1 0
              0 5
            ---------
              0 6 1 5
                1 2
              0 8
            0 4
            ---------
            0 5 5 3 5
这样我们就可以单独都对每一位进行相乘计算把结果存入相应的index中
'''
class Solution(object):
    def multiply(self, s1, s2):
        if not s1 or not s2 or s1 == '0' or s2 == '0':
            return '0'
        la = len(s1)
        lb = len(s2)
        res = [0]*(la+lb)
        for i in range(la-1,-1,-1):
            for j in range(lb-1,-1,-1):
                tmp = int(s1[i])*int(s2[j])
                res[i+j+1]+=(tmp%10)
                res[i+j]+=(tmp/10)
                if res[i+j+1] >= 10:
                    res[i+j] += 1
                    res[i+j+1] %= 10
                if res[i+j] >= 10:
                    res[i+j-1] += 1
                    res[i+j] %= 10
        return (''.join(str(ch) for ch in res)).lstrip('0')  # 剔除左边多余的0





# 等价问题：大数乘法  也是需要一位位的乘，放进数组里
class Solution(object):
    def BigMultiply(self, s1, s2):
        s1 = str(s1)
        s2 = str(s2)
        la = len(s1)
        lb = len(s2)
        res = [0]*(la+lb)
        for i in range(la-1,-1,-1):
            for j in range(lb-1,-1,-1):
                tmp = int(s1[i])*int(s2[j])
                res[i+j+1]+=(tmp%10)
                res[i+j]+=(tmp/10)
                if res[i+j+1] >= 10:
                    res[i+j] += 1
                    res[i+j+1] %= 10
                if res[i+j] >= 10:
                    res[i+j-1] += 1
                    res[i+j] %= 10
        return int((''.join(str(ch) for ch in res)).lstrip('0'))

s = Solution()
print s.BigMultiply(789634552, 32110)