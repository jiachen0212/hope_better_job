# coding=utf-8
# 复数乘法

class Solution(object):
    def complexNumberMultiply(self, a, b):
        # strip()函数: 删除字符串首尾指定char
        a = [int(i) for i in a.strip('i').split('+')]
        # 这样操作，就把a、b中的实虚部的值给剥开了!!! 很秀!!!
        b = [int(i) for i in b.strip('i').split('+')]
        ans = [a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]]
        return '{}+{}i'.format(*ans)

s = Solution()
res = s.complexNumberMultiply("2+-3i", "1+5i")
print(res)
