# coding=utf-8

# 1. 二分法
def mySqrt(x):
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


# 2. 牛顿法(梯度更新)
class Solution(object):
    def mySqrt(self, x):
        if x <= 1:
            return x
        x0 = x
        while(x0**2 - x) / (2*x0) > 1e-6:
            x0 = x0 - (x0**2 - x) / (2*x0)
        return int(x0) - 1

ans = mySqrt(8192)
print(ans)

s = Solution()
res = s.mySqrt(8192)
print res