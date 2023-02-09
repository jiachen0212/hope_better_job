# coding=utf-8
# 整数除法
# 左移动实现
class Solution(object):
    '''
每次都从2^0+2^1+...开始逼近
当快要接近被除数时, 从2^0+2^1+...  up >= down和up -= base的配合作用
    '''
    def divide(self, dividend, divisor):  # 被除数/除数
        divisor1 = abs(divisor)
        dividend1 = abs(dividend)
        if divisor == 0:
            return
        if dividend == 0:
            return 0
        ans, up, down = 0, dividend1, divisor1
        while up >= down:
            count = 1
            base = down
            while up > (base << 1):  # 左移*2
                count <<= 1   # 100 3 为栗, count: 1 2 4 8 16 32
                base <<= 1    # 3 6 12 24 48 96 192 出while循环
            ans += count    # ans=0+32    然后第二次是不进入内层循环 count=1 ans+=1 == 33
            up -= base      # up: 100 100-96=4

        # 除数被除数符号判断
        if divisor > 0 and dividend < 0:
            ans = -ans
        if divisor < 0 and dividend > 0:
            ans = -ans

        # 溢出修正
        if ans <= -2**31:
            return -2**31
        if ans > 2**31-1:
            return 2**31-1
        return ans

s = Solution()
print s.divide(7, -3)